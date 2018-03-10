#include <iostream>

using std::cout;
using std::endl;

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include "ufunc_utility.h"

namespace py = pybind11;

class Custom {
 public:
  using Self = Custom;

  Custom() {}
  Custom(double value) : value_{value} {}
  Custom(const Custom&) = default;
  Custom& operator=(const Custom&) = default;

  double operator==(const Self& rhs) const { return value_ + rhs.value_; }
  double value() const { return value_; }

  static double equal(const Self& lhs, const Self& rhs) {
    return lhs == rhs;
  }

 private:
  double value_{};
};

void module(py::module m) {}



int main() {
  py::scoped_interpreter guard;

  py::module m("__main__");

  using Class = Custom;
  py::class_<Class> cls(m, "Custom");
  cls
      .def(py::init<double>())
      .def(py::self == Class{})
      .def("value", &Class::value);

  py::exec(R"""(
a = Custom(1)
print(a == a)
)""");

  // Register thing.
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
      0,
      NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM, /* hasobject */
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
    auto& obj = *(Class*)in;
    py::object py_obj(in);
    obj = py::cast<Class>(py_obj);
    return 0;
  };
  Py_TYPE(&npyrational_descr) = &PyArrayDescr_Type;
  int npy_rational = PyArray_RegisterDataType(&npyrational_descr);
  cls.attr("dtype") = py::object(&npyrational_descr);

  auto ufunc = [&numpy](const char* name) {
    return (PyUFuncObject*)numpy.attr(name).ptr();
  };

  BinaryUFunc<Class, Class, double, Class::equal>::Register(ufunc("equal"));

  return 0;
}
