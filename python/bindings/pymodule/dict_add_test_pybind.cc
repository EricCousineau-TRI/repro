#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int main() {
  py::scoped_interpreter guard;

  py::dict out;
  out["a"] = 1;
  py::dict sub;
  sub["b"] = 10;
  out.attr("update")(sub);
  py::print(out);
}
