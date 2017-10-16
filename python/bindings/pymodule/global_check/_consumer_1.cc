#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "producer.h"

namespace py = pybind11;
using std::string;

namespace global_check {

std::pair<string, double> Consumer1(double value) {
  return Producer(value);
}

std::pair<string, double> Consumer1B(double value) {
  return ProducerB(value);
}

PYBIND11_MODULE(_consumer_1, m) {
  m.doc() = "Simple check on inheritance";
  m.def("consume", &Consumer1);
  m.def("consume_b", &Consumer1B);
}

}  // namespace global_check
