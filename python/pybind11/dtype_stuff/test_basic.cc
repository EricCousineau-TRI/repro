#include <cmath>

#include <map>
#include <iostream>

using std::pow;
using std::cout;
using std::endl;

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/ndarraytypes.h>

#include "ufunc_utility.h"
#include "ufunc_op.h"

namespace py = pybind11;

class Custom {
public:
    using Self = Custom;

    Custom() {}
    Custom(double value) : value_{value} {}
    Custom(const Custom&) = default;
    Custom& operator=(const Custom&) = default;
    double value() const { return value_; }

    operator double() const { return value_; }

    Self operator==(const Self& rhs) const { return value_ + rhs.value_; }
    Self operator<(const Self& rhs) const { return value_ * 10 * rhs.value_; }
    Custom operator*(const Custom& rhs) const {
        return value_ * rhs.value_;
    }
    Custom operator-() const { return -value_; }

    Self& operator+=(const Self& rhs) { value_ += rhs.value_; return *this; }
    Self operator+(const Self& rhs) const { return Self(*this) += rhs.value_; }
    Self operator-(const Self& rhs) const { return value_ - rhs.value_; }

private:
    double value_{};
};

Custom pow(Custom a, Custom b) {
  return pow(a.value(), b.value());
}

void module(py::module m) {}

int npy_custom{-1};

namespace pybind11 { namespace detail {

template <>
struct npy_format_descriptor<py::object> {
    static pybind11::dtype dtype() {
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(NPY_OBJECT))
            return reinterpret_borrow<pybind11::dtype>(ptr);
        pybind11_fail("Unsupported buffer format!");
    }
};

template <>
struct npy_format_descriptor<Custom> {
    static pybind11::dtype dtype() {
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(npy_custom))
            return reinterpret_borrow<pybind11::dtype>(ptr);
        pybind11_fail("Unsupported buffer format!");
    }
};

} }  // namespace detail } namespace pybind11

template <typename T>
std::type_index index_of() {
  return std::type_index(typeid(T));
}

auto& cls_map() {
  static std::map<std::type_index, py::handle> value;
  return value;
}

template <typename T>
static py::handle get_class() {
  return cls_map().at(index_of<T>());
}

template <typename Class>
struct DTypeObject {
  PyObject_HEAD
  Class value;

  static Class* load_raw(py::handle src) {
    DTypeObject* obj = (DTypeObject*)src.ptr();
    return &obj->value;
  }

  static ClassObject* alloc_py() {
    auto cls = get_class<Class>();
    PyTypeObject* cls_raw = (PyTypeObject*)cls.ptr();
    return (ClassObject*)cls_raw->tp_alloc((PyTypeObject*)cls.ptr(), 0);
  }

  static PyObject* tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    return (PyObject*)alloc_py();
  }
};

// Value.
template <typename Class, bool is_pointer = false>
struct dtype_arg {
  using ClassObject = DTypeObject<Class>;
  dtype_arg(py::handle h)
    : h_(h), is_py_{true} {}
  dtype_arg(const T& obj)
    : obj_(obj), is_py_{false} {}

  bool load(bool convert) {
    assert(is_py_);
    auto cls = get_class<Class>();
    if (!py::isinstance(src, cls)) {
      if (convert) {
        throw py::cast_error("not implemented");
      } else {
        throw py::cast_error("Must be of the same type");
      }
    } else {
      value_ = *ClassObject::load_raw(h_);
      return true;
    }
  }
  Class& value() const {
    return value_;
  }
  Class* ptr() const {
    return value_ptr_;
  }
  py::object py() const {
    if (!h_) {
      ClassObject* obj = ClassObject::alloc_py();
      obj->value = value_;
    }
    return h_;
  }
 private:
  bool is_py_{};
  py::handle h_;
  T* value_ptr_;
};

template <typename Class>
struct dtype_arg_caster {
  // Using this structure because `intrinsic_t` masks our abilities to natively
  // use pointers when using `make_caster` with the temporary return check.
  // See fork, modded func `cast_is_known_safe`.
  using Arg = dtype_arg<Class>;
  using ClassObject = typename DType::ClassObject;
  static py::handle cast(Arg& src, py::return_value_policy, py::handle) {
    return src.py().release();
  }
  bool load(py::handle src, bool convert) {
    value_ = src;
    return value_.load(convert);
  }
  template <typename T_> using cast_op_type = pybind11::detail::movable_cast_op_type<T_>;

  operator Arg&() { return value_; }
  Arg value_;
};

template <typename Class>

// Following `pybind11/detail/init.h`
template <typename... Args>
struct dtype_init_factory {
  template <typename PyClass, typename... Extra>
  void execute(PyClass& cl, const Extra&... extra) {
    using Class = typename PyClass::Class;
    using ClassObject = typename PyClass::ClassObject;
    // Do not construct this with the name `__init__` as pybind will try to
    // take over the init setup.
    cl.def("_dtype_init", [](dtype_arg<Class*> self, Args... args) {
      // Old-style. No factories for now.
      new (self.value()) Class(std::forward<Args>(args)...);
    });
    if (!py::hasattr(cl, "__init__")) {
      py::handle h = cl;
      auto init = cl.attr("_dtype_init");
      auto func = py::cpp_function(
          [init](py::handle self, py::args args, py::kwargs kwargs) {
            // Dispatch.
            self.attr("_dtype_init")(*args, **kwargs);
          }, py::is_method(h));
      cl.attr("__init__") = func;
    }
  }
};

template <typename... Args>
dtype_init_factory<Args...> dtype_init() { return {}; }

template <typename Class_>
class dtype_class : public py::class_<Class_> {
 public:
  using Base = py::class_<Class_>;
  using Class = Class_;
  using ClassObject = DTypeObject<Class>;
  // https://stackoverflow.com/a/12505371/7829525
  
  dtype_class(py::handle scope, const char* name) : Base(py::none()) {
    auto heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
    PY_ASSERT_EX(heap_type, "yar");
    heap_type->ht_name = py::str(name).release().ptr();
    // It's painful to inherit from `np.generic`, because it has no `tp_new`.
    auto& ClassObject_Type = heap_type->ht_type;
    ClassObject_Type.tp_base = &PyGenericArrType_Type;
    // Define other things.
    ClassObject_Type.tp_new = &ClassObject::tp_new;
    ClassObject_Type.tp_name = name;  // Er... scope?
    ClassObject_Type.tp_basicsize = sizeof(ClassObject);
    // ClassObject_Type.tp_getattro = PyObject_GenericGetAttr;
    // ClassObject_Type.tp_setattro = PyObject_GenericSetAttr;
    ClassObject_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
    // ClassObject_Type.tp_dictoffset = offsetof(ClassObject, dict);
    ClassObject_Type.tp_doc = "Stuff.";
    PY_ASSERT_EX(PyType_Ready(&ClassObject_Type) == 0, "");
    py::object* self = this;
    *self = py::reinterpret_borrow<py::object>(py::handle((PyObject*)&ClassObject_Type));
    scope.attr(name) = *self;
    cls_map()[index_of<Class>()] = *self;
  }

  template <typename ... Args, typename... Extra>
  dtype_class &def_dtype(dtype_init_factory<Args...>&& init, const Extra&... extra) {
    std::move(init).execute(*this, extra...);
    return *this;
  }
};

namespace pybind11 { namespace detail {

template <>
struct type_caster<Custom> : public dtype_caster<Custom> {};

// template <typename T>
// struct cast_is_known_safe<T,
//     enable_if_t<std::is_same<dtype_caster_ptr<std::remove_pointer_t<T>>, make_caster<T>>::value>>
//     : public std::true_type {};

} } // namespace pybind11 { namespace detail {

int main() {
    py::scoped_interpreter guard;

using Class = Custom;

    _import_array();
    _import_umath();
    py::module numpy = py::module::import("numpy");
    py::module m("__main__");
    py::dict md = m.attr("__dict__");
    py::dict locals;

    // static_assert(py::detail::cast_is_known_safe<Custom*>::value, "Yoew");

    dtype_class<Custom> py_type(m, "Custom");
    // Do not define `__init__`. Rather, use a custom thing.
    py_type
        .def_dtype(dtype_init<double>())
        // .def(py::self == Class{})
        // .def(py::self * Class{})
        // .def("value", &Class::value)
        .def("__repr__", [](const Class* self) {
            return py::str("_Custom({})").format(self->value());
        });

    py::exec(R"""(
print(Custom)
#print(Custom(1))
)""", md, md);

#if 0
    typedef struct { char c; Class r; } align_test;
    static PyArray_ArrFuncs arrfuncs;
    static PyArray_Descr descr = {
        PyObject_HEAD_INIT(0)
        &ClassObject_Type,                /* typeobj */
        'V',                    /* kind (V = arbitrary) */
        'r',                    /* type */
        '=',                    /* byteorder */
        /*
         * For now, we need NPY_NEEDS_PYAPI in order to make numpy detect our
         * exceptions.  This isn't technically necessary,
         * since we're careful about thread safety, and hopefully future
         * versions of numpy will recognize that.
         */
        NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, /* flags */
        0,                      /* type_num */
        sizeof(Class),       /* elsize */
        offsetof(align_test,r), /* alignment */
        0,                      /* subarray */
        0,                      /* fields */
        0,                      /* names */
        &arrfuncs,  /* f */
    };

    py::class_<Class> cls(m, "_Custom");
//        py::handle((PyObject*)&ClassObject_Type));
    cls
        .def(py::init([](double x) {
            return new Custom(x);
        }))

    static auto from_py = [](py::handle h) {
      py::print(py::str("yar eadf: {}").format(h.get_type()));
        // return Class(10);
      py::object yar = py::module::import("__main__").attr("Custom").attr("maybe_value");
     return *yar(h).cast<Class*>();
    };
    static auto to_py = [](const Class* obj) {
      py::object yar = py::module::import("__main__").attr("_Custom");
      return yar(py::cast(*obj)).release().ptr();
    };

    PyArray_InitArrFuncs(&arrfuncs);
    // https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
    arrfuncs.getitem = [](void* in, void* arr) -> PyObject* {
        return to_py((const Class*)in);
    };
    arrfuncs.setitem = [](PyObject* in, void* out, void* arr) {
        *(Class*)out = from_py(in);
        return 0;
    };
    arrfuncs.copyswap = [](void* dst, void* src, int swap, void* arr) {
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
    npy_custom = PyArray_RegisterDataType(&descr);
     py_type.attr("dtype") = py::reinterpret_borrow<py::object>(
         py::handle((PyObject*)&descr));

    py::exec(R"""(
c = Custom(1)
print(dir(c))
print(repr(c))
)""", m.attr("__dict__"), m.attr("__dict__"));
    return 0;

//     PyDict_SetItemString(ClassObject_Type.tp_dict, "blergh", Py_None);

//     py::exec(R"""(
// _Custom.junk = 2
// print(_Custom.blergh)
// print(_Custom.__dict__)
// def _c_init(self, x):
//     self.x = x
// _Custom.__init__ = _c_init
// _Custom.__repr__ = lambda self: "blerg"
// )""", m.attr("__dict__"));

    using Unary = type_pack<Class>;
    using Binary = type_pack<Class, Class>;
    // Arithmetic.
    maybe_ufunc<check_add, Class>(Binary{});
    maybe_ufunc<check_negative, Class>(Unary{});
    maybe_ufunc<check_multiply, Class>(Binary{});
    maybe_ufunc<check_divide, Class>(Binary{});
    maybe_ufunc<check_power, Class>(Binary{});
    maybe_ufunc<check_subtract, Class>(Binary{});
    // Comparison.
    maybe_ufunc<check_greater, Class>(Binary{});
    maybe_ufunc<check_greater_equal, Class>(Binary{});
    maybe_ufunc<check_less, Class>(Binary{});
    maybe_ufunc<check_less_equal, Class>(Binary{});
    maybe_ufunc<check_equal, Class>(Binary{});
    maybe_ufunc<check_not_equal, Class>(Binary{});

    // Casting.
    maybe_cast<Class, double>();
    maybe_cast<double, Class>();
    add_cast<Class, py::object>([](const Class& obj) {
      return py::cast(obj);
    });
    add_cast<py::object, Class>([](py::object obj) {
      return py::cast<Class>(obj);
    });
    // - _zerofill? What do I need?
    add_cast<int, Class>([](int x) {
      return Class{static_cast<double>(x)};
    });
    add_cast<long, Class>([](long x) {
      return Class{static_cast<double>(x)};
    });


    py::str file = "python/pybind11/dtype_stuff/test_basic.py";
    py::print(file);
    m.attr("__file__") = file;
    py::eval_file(file);

    py::exec(R"""(
x = np.array([Custom(1)])
)""");
#endif // 0

    return 0;
}
