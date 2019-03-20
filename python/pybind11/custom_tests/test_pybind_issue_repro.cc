// https://github.com/pybind/pybind11/issues/1729

#include <string>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

class Test {
    public:
        using Ptr = std::shared_ptr<Test>;
        using Data = std::vector<std::string>;

        Test(Data data_)
            : data{std::move(data_)}{}

        const Data data;
};

void init_module(py::module m) {
  py::class_<Test, Test::Ptr>(m, "Test")
      .def(py::init<Test::Data>(), "data"_a)
      .def_readonly("data", &Test::data);
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
print(m.Test([]).data)
print(m.Test(["a", "b", "c"]).data)
)""");

  py::print("[ Done ]");

  return 0;
}
