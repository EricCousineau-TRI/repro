#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "producer.h"

namespace py = pybind11;
using std::string;

namespace global_check {

std::pair<string, double> Consumer2(double value) {
  return Producer(value);
}

PYBIND11_MODULE(_consumer_2, m) {
  // Check shared library separation.
  m.def("consume", &Consumer2);
}

}  // namespace global_check
