#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "producer.h"

namespace py = pybind11;
using std::string;

namespace global_check {

string Consumer2(double value) {
  return Producer(value);
}

PYBIND11_PLUGIN(_consumer_2) {
  py::module m("_consumer_2", "");
  m.def("do_stuff_2", &Consumer2);
  return m.ptr();
}

}  // namespace global_check
