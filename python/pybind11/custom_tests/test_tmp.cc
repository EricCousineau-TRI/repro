// For: https://github.com/pybind/pybind11/issues/1640

#include <Eigen/Dense>

#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using rvp = py::return_value_policy;

class Original {
 public:
  virtual ~Original() {}

  int stuff() const { return 1; }
};

class PyOriginal : public Original {};

void init_module(py::module m) {
  py::class_<Original, PyOriginal>(m, "Original")
    .def(py::init())
    .def("stuff", &Original::stuff);
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
Original = m.Original

class ExtendA(Original):
    def extra_a(self):
        return 2


class ExtendB(Original):
    def extra_b(self):
        return 3


def main():
    # N.B. These do work if the creation is changed to `ExtendA()` / 
    # `ExtendB()`.
    obj = Original()
    obj.__class__ = ExtendA
    assert obj.stuff() == 1
    assert isinstance(obj, Original)
    assert obj.extra_a() == 2
    assert type(obj) == ExtendA

    obj = Original()
    obj.__class__ = ExtendB
    assert obj.stuff() == 1
    assert isinstance(obj, Original)
    assert obj.extra_b() == 3
    assert type(obj) == ExtendB

    print("[ Done ]")


assert __name__ == "__main__"
main()
)""");

  py::print("[ Done ]");

  return 0;
}

/*
Output:

terminate called after throwing an instance of 'pybind11::error_already_set'
  what():  TypeError: __class__ assignment: 'ExtendB' deallocator differs from 'test_module.Original'

*/