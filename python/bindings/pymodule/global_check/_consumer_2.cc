#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "producer.h"

namespace py = pybind11;
using std::string;

namespace global_check {

std::pair<string, double> Consumer2(double value) {
  return Producer(value);
}

std::pair<string, double> Consumer2B(double value) {
  return ProducerB(value);
}

PYBIND11_MODULE(_consumer_2, m) {
  m.doc() = "Simple check on inheritance";
  m.def("consume", &Consumer2);
  m.def("consume_b", &Consumer2B);
}

}  // namespace global_check
