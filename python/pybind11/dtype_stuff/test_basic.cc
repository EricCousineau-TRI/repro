#include <cmath>

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

// Could increase param count?
#define CHECK_EXPR(struct_name, name, expr, lambda_expr) \
    template <typename A_, typename B = A_> \
    struct struct_name { \
      template <typename A = A_> \
      static std::true_type check(decltype(expr)*); \
      template <typename> \
      static std::false_type check(...); \
      static constexpr bool value = decltype(check<A_>(nullptr))::value; \
      template <typename A = A_> \
      static auto get_lambda() { return lambda_expr; } \
      static const char* get_name() { return name; } \
    };

// https://docs.scipy.org/doc/numpy/reference/routines.math.html
CHECK_EXPR(check_add, "add", A{} + B{},
           [](const A& a, const B& b) { return a + b; });
CHECK_EXPR(check_negative, "negative", -A{},
           [](const A& a) { return -a; });
CHECK_EXPR(check_multiply, "multiply", A{} * B{},
           [](const A& a, const B& b) { return a * b; });
CHECK_EXPR(check_divide, "divide", A{} / B{},
           [](const A& a, const B& b) { return a / b; });
// TODO(eric.cousineau): Figger out non-operator things...
CHECK_EXPR(check_power, "power", pow(A{}, B{}),
           [](const A& a, const B& b) { return pow(a, b); });
CHECK_EXPR(check_subtract, "subtract", A{} - B{},
           [](const A& a, const B& b) { return a - b; });
// https://docs.scipy.org/doc/numpy/reference/routines.logic.html
CHECK_EXPR(check_greater, "greater", A{} > B{},
           [](const A& a, const B& b) { return a > b; });
CHECK_EXPR(check_greater_equal, "greater_equal", A{} >= B{},
           [](const A& a, const B& b) { return a >= b; });
CHECK_EXPR(check_less, "less", A{} < B{},
           [](const A& a, const B& b) { return a < b; });
CHECK_EXPR(check_less_equal, "less_equal", A{} <= B{},
           [](const A& a, const B& b) { return a <= b; });
CHECK_EXPR(check_equal, "equal", A{} == B{},
           [](const A& a, const B& b) { return a == b; });
CHECK_EXPR(check_not_equal, "not_equal", A{} != B{},
           [](const A& a, const B& b) { return a != b; });
// Casting.
CHECK_EXPR(check_cast, "", static_cast<B>(A{}),
           [](const A& a) { return static_cast<B>(a); });

template <typename Result, typename Func>
void run_if(Func&& func) {
  using Pack = type_pack<Result>;
  type_visit_impl<visit_with_default, Func>::
      template runner<Pack, Result::value>::run(func);
}

PyUFuncObject* get_py_ufunc(const char* name) {
  py::module numpy = py::module::import("numpy");
  return (PyUFuncObject*)numpy.attr(name).ptr();
}

template <template <typename...> class Check, typename Class, typename ... Args>
void maybe_ufunc(type_pack<Args...> = {}) {
  using Result = typename type_pack<Args...>::template bind<Check>;
  constexpr int N = sizeof...(Args);
  auto defer = [](auto) {
    RegisterUFunc<Class>(
        get_py_ufunc(Result::get_name()), Result::get_lambda(), const_int<N>{});
  };
  run_if<Result>(defer);
}

template <typename From, typename To, typename Func>
void add_cast(Func&& func, type_pack<From, To> = {}) {
  auto* from = PyArray_DescrFromType(dtype_num<From>());
  int to = dtype_num<To>();
  static auto cast_lambda = func;
  auto cast_func = [](
        void* from_, void* to_, npy_intp n,
        void* fromarr, void* toarr) {
      const From* from = (From*)from_;
      To* to = (To*)to_;
      for (npy_intp i = 0; i < n; i++)
          to[i] = cast_lambda(from[i]);
  };
  PY_ASSERT_EX(
      PyArray_RegisterCastFunc(from, to, cast_func) >= 0,
      "Cannot register cast");
  PY_ASSERT_EX(
      PyArray_RegisterCanCast(from, to, NPY_NOSCALAR) >= 0,
      "Cannot register castability");
}

template <typename From, typename To>
void maybe_cast(type_pack<From, To> = {}) {
  using Result = check_cast<From, To>;
  auto defer = [](auto) {
    add_cast<From, To>(Result::get_lambda());
  };
  run_if<Result>(defer);
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



// static PyNumberMethods custom_as_number = {
//     0,          /* nb_add */
//     0,     /* nb_subtract */
//     0,     /* nb_multiply */
// #if PY_MAJOR_VERSION < 3
//     0,       /* nb_divide */
// #endif
//     0,    /* nb_remainder */
//     0,                       /* nb_divmod */
//     0,                       /* nb_power */
//     0,     /* nb_negative */
//     0,     /* nb_positive */
//     0,     /* nb_absolute */
//     0,      /* nb_nonzero */
//     0,                       /* nb_invert */
//     0,                       /* nb_lshift */
//     0,                       /* nb_rshift */
//     0,                       /* nb_and */
//     0,                       /* nb_xor */
//     0,                       /* nb_or */
// #if PY_MAJOR_VERSION < 3
//     0,                       /* nb_coerce */
// #endif
//     0,          /* nb_int */
// #if PY_MAJOR_VERSION < 3
//     0,          /* nb_long */
// #else
//     0,                       /* reserved */
// #endif
//     0,        /* nb_float */
// #if PY_MAJOR_VERSION < 3
//     0,                       /* nb_oct */
//     0,                       /* nb_hex */
// #endif

//     0,                       /* nb_inplace_add */
//     0,                       /* nb_inplace_subtract */
//     0,                       /* nb_inplace_multiply */
// #if PY_MAJOR_VERSION < 3
//     0,                       /* nb_inplace_divide */
// #endif
//     0,                       /* nb_inplace_remainder */
//     0,                       /* nb_inplace_power */
//     0,                       /* nb_inplace_lshift */
//     0,                       /* nb_inplace_rshift */
//     0,                       /* nb_inplace_and */
//     0,                       /* nb_inplace_xor */
//     0,                       /* nb_inplace_or */

//     0, /* nb_floor_divide */
//     0,       /* nb_true_divide */
//     0,                       /* nb_inplace_floor_divide */
//     0,                       /* nb_inplace_true_divide */
//     0,                       /* nb_index */
// };

typedef struct {
    PyObject_HEAD
    Custom r;
} PyCustom;

extern PyTypeObject PyCustom_Type;

static PyObject*
PyCustom_FromCustom(Custom x) {
    PyCustom* p = (PyCustom*)PyCustom_Type.tp_alloc(&PyCustom_Type,0);
    if (p) {
        p->r = x;
    }
    return (PyObject*)p;
}

static PyObject*
custom_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    return PyCustom_FromCustom(Custom{});
}

PyTypeObject PyCustom_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size */
#endif
    "Custom",                               /* tp_name */
    sizeof(PyCustom),                       /* tp_basicsize */
    0,                                        /* tp_itemsize */
    0,                                        /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                        /* tp_reserved */
#else
    0,                                        /* tp_compare */
#endif
    0,                          /* tp_repr */
    0,                    /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                                        /* tp_call */
    0,                           /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Minimal wrap for numpy",       /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                   /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    0,                        /* tp_getset */
    0, /* tp_base - To be set when numpy is imported */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    custom_new,                           /* tp_new */
    0,                                        /* tp_free */
    0,                                        /* tp_is_gc */
    0,                                        /* tp_bases */
    0,                                        /* tp_mro */
    0,                                        /* tp_cache */
    0,                                        /* tp_subclasses */
    0,                                        /* tp_weaklist */
    0,                                        /* tp_del */
    0,                                        /* tp_version_tag */
};

int main() {
    py::scoped_interpreter guard;


    _import_array();
    _import_umath();
    py::module numpy = py::module::import("numpy");
    py::module m("__main__");

    using Class = Custom;
    py::class_<Class> cls(m, "Custom");
    cls
        .def(py::init<double>())
        .def(py::self == Class{})
        .def(py::self * Class{})
        .def("value", &Class::value)
        .def("__repr__", [](const Class* self) {
            return py::str("Custom({})").format(self->value());
        });

    // Register thing.
    auto py_type = &PyCustom_Type;
    PyCustom_Type.tp_base = &PyGenericArrType_Type;
    PY_ASSERT_EX(PyType_Ready(&PyCustom_Type) >= 0, "Crap");

    typedef struct { char c; Class r; } align_test;

    static PyArray_ArrFuncs arrfuncs;
    
    static PyArray_Descr descr = {
        PyObject_HEAD_INIT(0)
        py_type,                /* typeobj */
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

    PyArray_InitArrFuncs(&arrfuncs);
    // https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
    arrfuncs.getitem = [](void* in, void* arr) -> PyObject* {
        return py::cast(*(const Class*)in).release().ptr();
    };
    arrfuncs.setitem = [](PyObject* in, void* out, void* arr) {
        *(Class*)out = *py::handle(in).cast<Class*>();
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
    // For `zeros`, `ones`, etc.
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
    cls.attr("dtype") = py::reinterpret_borrow<py::object>(
        py::handle((PyObject*)&descr));

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

    return 0;
}
