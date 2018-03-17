#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int main() {
  py::scoped_interpreter guard;

  // Option 1
  py::module np = py::module::import("numpy");
  py::print(np.attr("float32")(10.2));
  // Option 2
  py::object float32 = py::dtype::of<float>().attr("type");
  py::print(float32(10.2));

  return 0;
}
