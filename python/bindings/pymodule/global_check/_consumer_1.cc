#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "producer.h"

namespace py = pybind11;
using std::string;

namespace global_check {

string Consumer1(double value) {
  return Producer(value);
}

PYBIND11_PLUGIN(_consumer_1) {
  py::module m("_consumer_1", "Simple check on inheritance");
  m.def("do_stuff_1", &Consumer1);
  return m.ptr();
}

}  // namespace global_check
