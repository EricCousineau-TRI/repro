#include <iostream>

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

  Self operator==(const Self& rhs) const { return value_ + rhs.value_; }
  double value() const { return value_; }

  static Self equal(const Self& lhs, const Self& rhs) {
    return lhs == rhs;
  }

 private:
  double value_{};
};

void module(py::module m) {}

int npy_rational{-1};

namespace pybind11 { namespace detail {

template <>
struct npy_format_descriptor<Custom> {
  static const int& value;
};

const int& npy_format_descriptor<Custom>::value = npy_rational;

} }  // namespace detail } namespace pybind11

static NPY_INLINE void
byteswap(npy_int32* x) {
    char* p = (char*)x;
    size_t i;
    for (i = 0; i < sizeof(*x)/2; i++) {
        size_t j = sizeof(*x)-1-i;
        char t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
}

int main() {
  py::scoped_interpreter guard;

  py::module m("__main__");

  using Class = Custom;
  py::class_<Class> cls(m, "Custom");
  cls
      .def(py::init<double>())
      .def(py::self == Class{})
      .def("value", &Class::value)
      .def("__repr__", [](const Class* self) {
        return py::str("Custom({})").format(self->value());
      });

  py::exec(R"""(
a = Custom(1)
print(a == a)
import sys; sys.stdout.flush();
)""");

  // Register thing.

  _import_array();
  _import_umath();

  py::module numpy = py::module::import("numpy");
  auto py_type = (PyTypeObject*)cls.ptr();

  typedef struct { char c; Class r; } align_test;

  static PyArray_ArrFuncs npyrational_arrfuncs;
  static PyArray_Descr npyrational_descr = {
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
      &npyrational_arrfuncs,  /* f */
  };

  PyArray_InitArrFuncs(&npyrational_arrfuncs);
  npyrational_arrfuncs.getitem = [](void* in, void* arr) -> PyObject* {
    const auto& obj = *(const Class*)in;
    return py::cast(obj, py::return_value_policy::copy).release().ptr();
  };
  npyrational_arrfuncs.setitem = [](PyObject* in, void* out, void* arr) {
    *(Class*)out = *py::handle(in).cast<Class*>();
    return 0;
  };
  npyrational_arrfuncs.copyswap = [](void* dst, void* src, int swap, void* arr) {
      Class* r_dst = (Class*)dst;
      Class* r_src = (Class*)src;
      if (!src) return;
      if (swap) {
          std::swap(*r_dst, *r_src);
      } else {
        *r_dst = *r_src;
      }
  };
  Py_TYPE(&npyrational_descr) = &PyArrayDescr_Type;
  npy_rational = PyArray_RegisterDataType(&npyrational_descr);
  cls.attr("dtype") = py::reinterpret_borrow<py::object>(
      py::handle((PyObject*)&npyrational_descr));

  auto ufunc = [&numpy](const char* name) {
    return (PyUFuncObject*)numpy.attr(name).ptr();
  };

  BinaryUFunc<Class, Class, Class, Class::equal>::Register(ufunc("equal"));

  py::exec(R"""(
import numpy as np
print(np.dtype(Custom))
print(Custom.dtype)
print(type(Custom.dtype))

av = np.array([a, a], dtype=Custom)
print(a)
print(av)
print(av == av)
sys.stdout.flush()
)""");

  return 0;
}
