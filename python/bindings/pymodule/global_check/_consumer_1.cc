#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "producer.h"

namespace py = pybind11;
using std::string;

namespace global_check {

std::pair<string, double> consume_linkstatic(double value) {
  return Producer(value);
}

std::pair<string, double> consume_linkshared(double value) {
  return ProducerB(value);
}

PYBIND11_MODULE(_consumer_1, m) {
  m.doc() = "Linking tests";
  m.def("consume_linkstatic", &consume_linkstatic);
  m.def("consume_linkshared", &consume_linkshared);
}

}  // namespace global_check
