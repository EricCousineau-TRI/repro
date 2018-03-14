/**
@file
Provides mechanism to make user-defined NumPy dtypes.
Note that these are not "custom" dtypes (e.g. records); these are actually
new scalar types.
*/

#pragma once

#include <map>

#include <pybind11/pybind11.h>

#include "cpp/wrap_function.h"

#include "ufunc_utility.h"
#include "ufunc_op.h"

namespace py = pybind11;

struct dtype_info {
  py::handle cls;
  int dtype_num{-1};
  std::map<void*, PyObject*> instance_to_py;

 private:
  static auto& cls_map() {
    static std::map<std::type_index, dtype_info> value;
    return value;
  }

 public:
  template <typename T>
  static dtype_info& get_mutable_entry(bool is_new = false) {
    // TODO: implement is_new
    return cls_map()[std::type_index(typeid(T))];
  }

  template <typename T>
  static const dtype_info& get_entry() {
    return cls_map().at(std::type_index(typeid(T)));
  }
};

template <typename Class>
struct dtype_py_object {
  PyObject_HEAD
  // TODO(eric.cousineau): Reduce the number of temporaries. To construct one
  // item, three temporaries are generated. Consider storing (unique) pointer
  // here.
  Class value;

  static Class* load_raw(PyObject* src) {
    dtype_py_object* obj = (dtype_py_object*)src;
    return &obj->value;
  }

  static dtype_py_object* alloc_py() {
    auto cls = dtype_info::get_entry<Class>().cls;
    PyTypeObject* cls_raw = (PyTypeObject*)cls.ptr();
    return (dtype_py_object*)cls_raw->tp_alloc((PyTypeObject*)cls.ptr(), 0);
  }

  static PyObject* tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    // N.B. `__init__` should call the in-place constructor.
    auto obj = alloc_py();
    // // Register.
    auto& entry = dtype_info::get_mutable_entry<Class>();
    entry.instance_to_py[&obj->value] = (PyObject*)obj;
    return (PyObject*)obj;
  }

  static void tp_dealloc(PyObject* self) {
    Class* value = load_raw(self);
    // Call destructor.
    value->~Class();
    // Deregister.
    auto& entry = dtype_info::get_mutable_entry<Class>();
    entry.instance_to_py.erase(value);
  }

  static py::object find_existing(const Class* value) {
    auto& entry = dtype_info::get_entry<Class>();
    auto it = entry.instance_to_py.find((void*)value);
    if (it == entry.instance_to_py.end())
      return {};
    else {
      return py::reinterpret_borrow<py::object>(it->second);
    }
  }
};

template <typename Class>
struct dtype_caster {
  static constexpr auto name = py::detail::_<Class>();
  // Using this structure because `intrinsic_t` masks our abilities to natively
  // use pointers when using `make_caster` with the temporary return check.
  // See fork, modded func `cast_is_known_safe`.
  using DTypePyObject = dtype_py_object<Class>;
  static py::handle cast(const Class& src, py::return_value_policy, py::handle) {
    py::object h = DTypePyObject::find_existing(&src);
    // TODO(eric.cousineau): Handle parenting?
    if (!h) {
      // Make new instance.
      DTypePyObject* obj = DTypePyObject::alloc_py();
      obj->value = src;
      h = py::reinterpret_borrow<py::object>((PyObject*)obj);
      return h.release();
    }
    return h.release();
  }

  static py::handle cast(const Class* src, py::return_value_policy, py::handle) {
    py::object h = DTypePyObject::find_existing(src);
    if (h) {
      return h.release();
    } else {
      throw py::cast_error("Cannot find existing instance");
    }
  }

  bool load(py::handle src, bool convert) {
    auto cls = dtype_info::get_entry<Class>().cls;
    py::object obj;
    if (!py::isinstance(src, cls)) {
      if (convert) {
        // Just try to call it.
        // TODO(eric.cousineau): Take out the Python middle man?
        // Use registered ufuncs? See how `implicitly_convertible` is
        // implemented.
        obj = cls(src);
      } else {
        return false;
      }
    } else {
      obj = py::reinterpret_borrow<py::object>(src);
    }
    ptr_ = DTypePyObject::load_raw(obj.ptr());
    return true;
  }
  // Copy `type_caster_base`.
  template <typename T_> using cast_op_type =
      pybind11::detail::cast_op_type<T_>;

  operator Class&() { return *ptr_; }
  operator Class*() { return ptr_; }
  Class* ptr_{};
};

void init_numpy() {
  _import_array();
  _import_umath();
  py::module::import("numpy");
}

const char* get_ufunc_name(py::detail::op_id id) {
  using namespace py::detail;
  static std::map<op_id, const char*> m = {
    // https://docs.scipy.org/doc/numpy/reference/routines.math.html
    {op_add, "add"},
    {op_neg, "negative"},
    {op_mul, "multiply"},
    {op_div, "divide"},
    {op_pow, "power"},
    {op_sub, "subtract"},
    // https://docs.scipy.org/doc/numpy/reference/routines.logic.htmls
    {op_gt, "greater"},
    {op_ge, "greater_equal"},
    {op_lt, "less"},
    {op_le, "less_equal"},
    {op_eq, "equal"},
    {op_ne, "not_equal"},
    {op_bool, "nonzero"},
    {op_invert, "logical_not"}
    // TODO(eric.cousineau): Add something for junction-style logic?
  };
  return m.at(id);
}

template <typename Class_>
class dtype_user : public py::class_<Class_> {
 public:
  using Base = py::class_<Class_>;
  using Class = Class_;
  using DTypePyObject = dtype_py_object<Class>;
  // https://stackoverflow.com/a/12505371/7829525

  dtype_user(py::handle scope, const char* name) : Base(py::none()) {
    init_numpy();
    register_type(name);
    scope.attr(name) = self();
    auto& entry = dtype_info::get_mutable_entry<Class>(true);
    entry.cls = self();
    // Registry numpy type.
    // (Note that not registering the type will result in infinte recursion).
    entry.dtype_num = register_numpy();

    // Register default ufunc cast to `object`.
    this->def_ufunc_cast([](const Class& self) { return py::cast(self); });
    this->def_ufunc_cast([](py::object self) { return py::cast<Class>(self); });
  }

  ~dtype_user() {
    check();    
  }

  template <typename ... Args>
  dtype_user& def(const char* name, Args&&... args) {
    base().def(name, std::forward<Args>(args)...);
    return *this;
  }

  template <typename ... Args, typename... Extra>
  dtype_user& def(py::detail::initimpl::constructor<Args...>&& init, const Extra&... extra) {
    // Do not construct this with the name `__init__` as pybind will try to
    // take over the init setup.
    add_init([](Class* self, Args... args) {
      // Old-style. No factories for now.
      new (self) Class(std::forward<Args>(args)...);
    });
    return *this;
  }

  template <py::detail::op_id id, py::detail::op_type ot,
      typename L, typename R, typename... Extra>
  dtype_user& def_ufunc(
      const py::detail::op_<id, ot, L, R>& op, const Extra&... extra) {
    using op_ = py::detail::op_<id, ot, L, R>;
    using op_impl = typename op_::template info<dtype_user>::op;
    auto func = &op_impl::execute;
    const char* ufunc_name = get_ufunc_name(id);
    // Define operators.
    this->def(op_impl::name(), func, py::is_operator(), extra...);
    // Register ufunction.
    auto func_infer = detail::infer_function_info(func);
    using Func = decltype(func_infer);
    constexpr int N = Func::Args::size;
    RegisterUFunc<Class>(
        get_py_ufunc(ufunc_name), func, const_int<N>{});
    return *this;
  }

  // Nominal operator.
  template <py::detail::op_id id, py::detail::op_type ot,
      typename L, typename R, typename... Extra>
  dtype_user& def(
      const py::detail::op_<id, ot, L, R>& op, const Extra&... extra) {
    base().def(op, extra...);
    return *this;
  }

  template <typename Func_>
  dtype_user& def_ufunc_cast(Func_&& func) {
    auto func_infer = detail::infer_function_info(func);
    using Func = decltype(func_infer);
    using From = py::detail::intrinsic_t<typename Func::Args::template type_at<0>>;
    using To = py::detail::intrinsic_t<typename Func::Return>;
    add_cast<From, To>(func);
    return *this;
  }

 private:
  Base& base() { return *this; }
  py::object& self() { return *this; }
  const py::object& self() const { return *this; }

  void check() const {
    // This `dict` should indicate whether we've directly overridden methods.
    py::dict d = self().attr("__dict__");
    // Without these, numpy goes into infinite recursion. Haven't bothered to
    // figure out exactly why.
    if (!d.contains("__repr__")) {
      throw std::runtime_error("Class is missing explicit __repr__");
    }
    if (!d.contains("__str__")) {
      throw std::runtime_error("Class is missing explicit __str__");
    }
  }

  template <typename Func>
  void add_init(Func&& f) {
    this->def("_dtype_init", std::forward<Func>(f));
    // Ensure that this is called by a non-pybind11-instance `__init__`.
    py::dict d = self().attr("__dict__");
    if (!d.contains("__init__")) {
      auto init = self().attr("_dtype_init");
      auto func = py::cpp_function(
          [init](py::handle self, py::args args, py::kwargs kwargs) {
            // Dispatch.
            self.attr("_dtype_init")(*args, **kwargs);
          }, py::is_method(self()));
      self().attr("__init__") = func;
    }
  }

  void register_type(const char* name) {
    auto heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
    PY_ASSERT_EX(heap_type, "yar");
    heap_type->ht_name = py::str(name).release().ptr();
    // It's painful to inherit from `np.generic`, because it has no `tp_new`.
    auto& ClassObject_Type = heap_type->ht_type;
    ClassObject_Type.tp_base = &PyGenericArrType_Type;
    ClassObject_Type.tp_new = &DTypePyObject::tp_new;
    ClassObject_Type.tp_dealloc = &DTypePyObject::tp_dealloc;
    ClassObject_Type.tp_name = name;  // Er... scope?
    ClassObject_Type.tp_basicsize = sizeof(DTypePyObject);
    ClassObject_Type.tp_getset = 0;
    ClassObject_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
    PY_ASSERT_EX(PyType_Ready(&ClassObject_Type) == 0, "");
    self() = py::reinterpret_borrow<py::object>(py::handle((PyObject*)&ClassObject_Type));
  }

  int register_numpy() {
    // Adapted from `numpy/core/multiarrya/src/test_rational.c.src`.
    // Define NumPy description.
    auto type = (PyTypeObject*)self().ptr();
    typedef struct { char c; Class r; } align_test;
    static PyArray_ArrFuncs arrfuncs;
    static PyArray_Descr descr = {
        PyObject_HEAD_INIT(0)
        type,                   /* typeobj */
        'V',                    /* kind (V = arbitrary) */
        'r',                    /* type */
        '=',                    /* byteorder */
        NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, /* flags */
        0,                      /* type_num */
        sizeof(Class),          /* elsize */
        offsetof(align_test,r), /* alignment */
        0,                      /* subarray */
        0,                      /* fields */
        0,                      /* names */
        &arrfuncs,  /* f */
    };

    PyArray_InitArrFuncs(&arrfuncs);

    // https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
    arrfuncs.getitem = [](void* in, void* arr) -> PyObject* {
        auto item = (const Class*)in;
        return py::cast(*item).release().ptr();
    };
    arrfuncs.setitem = [](PyObject* in, void* out, void* arr) {
        dtype_caster<Class> caster;
        PY_ASSERT_EX(caster.load(in, true), "Could not convert");
        // Cut out the middle-man?
        *(Class*)out = caster;
        return 0;
    };
    arrfuncs.copyswap = [](void* dst, void* src, int swap, void* arr) {
        // TODO(eric.cousineau): Figure out actual purpose of this.
        if (!src) return;
        Class* r_dst = (Class*)dst;
        Class* r_src = (Class*)src;
        if (swap) {
            std::swap(*r_dst, *r_src);
        } else {
            *r_dst = *r_src;
        }
    };
    // - Test and ensure this doesn't overwrite our `equal` unfunc.
    arrfuncs.compare = [](const void* d1, const void* d2, void* arr) {
      return 0;
    };
    arrfuncs.fill = [](void* data_, npy_intp length, void* arr) {
      Class* data = (Class*)data_;
      Class delta = data[1] - data[0];
      Class r = data[1];
      npy_intp i;
      for (i = 2; i < length; i++) {
          r += delta;
          data[i] = r;
      }
      return 0;
    };
    arrfuncs.fillwithscalar = [](
            void* buffer_raw, npy_intp length, void* value_raw, void* arr) {
        const Class* value = (const Class*)value_raw;
        Class* buffer = (Class*)buffer_raw;
        for (int k = 0; k < length; k++) {
            buffer[k] = *value;
        }
        return 0;
    };
    Py_TYPE(&descr) = &PyArrayDescr_Type;
    int dtype_num = PyArray_RegisterDataType(&descr);
    self().attr("dtype") =
        py::reinterpret_borrow<py::object>(py::handle((PyObject*)&descr));
    return dtype_num;
  }
};

template <typename Class>
struct npy_format_descriptor_custom {
    static constexpr auto name = py::detail::_<Class>();
    static pybind11::dtype dtype() {
        int dtype_num = dtype_info::get_entry<Class>().dtype_num;
        if (auto ptr = py::detail::npy_api::get().PyArray_DescrFromType_(dtype_num))
            return py::reinterpret_borrow<pybind11::dtype>(ptr);
        py::pybind11_fail("Unsupported buffer format!");
    }
};

namespace pybind11 { namespace detail {

template <typename T>
struct cast_is_known_safe<T,
    enable_if_t<std::is_base_of<
        dtype_caster<intrinsic_t<T>>, make_caster<T>>::value>>
    : public std::true_type {};

template <>
struct npy_format_descriptor<py::object> {
    static pybind11::dtype dtype() {
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(NPY_OBJECT))
            return reinterpret_borrow<pybind11::dtype>(ptr);
        pybind11_fail("Unsupported buffer format!");
    }
};

} } // namespace pybind11 { namespace detail {
