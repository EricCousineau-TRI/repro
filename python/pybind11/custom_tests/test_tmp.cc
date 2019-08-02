#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_module(py::module m) {
  m.def("my_func", [](py::kwargs kwargs) {
    py::print(kwargs);
    const int a = kwargs["a"].cast<int>();
    const int b = kwargs["b"].cast<int>();
    py::print(py::str("a = {}").format(a));
    py::print(py::str("b = {}").format(b));
  });
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
m.my_func(a=1, b=2)
)""");

  py::print("[ Done ]");

  return 0;
}

/* Output:

{'a': 1, 'b': 2}
a = 1
b = 2

*/
