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

    // TODO(eric.cousineau): Ensure this class does not get cleared.
    py::exec(R"""(
import numpy as np

class CustomShim(np.generic):
    def __init__(self, x):
        if not isinstance(x, Custom):
            x = Custom(x)
        self.x = x

    def __repr__(self):
        return repr(self.x)
)""", m.attr("__dict__"), m.attr("__dict__"));

    // Register thing.
    auto py_type_py = m.attr("CustomShim");
    auto py_type = (PyTypeObject*)py_type_py.ptr();

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

    static auto from_py = [](py::handle h) {
      return *h.attr("x").cast<Class*>();
    };
    static auto to_py = [](const Class* obj) {
      py::object yar = py::module::import("__main__").attr("CustomShim");
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
    py_type_py.attr("dtype") = py::reinterpret_borrow<py::object>(
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
