#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

class Animal {
public:
    virtual ~Animal() { }
    std::string go_once() { return go(1); }; // <-- Attempting this
    virtual std::string go(int n_times) = 0;
};

class PyAnimal : public Animal {
public:
    /* Inherit the constructors */
    using Animal::Animal;

    /* Trampoline (need one for each virtual function) */
    std::string go(int n_times) override {
        PYBIND11_OVERLOAD_PURE(
            std::string, /* Return type */
            Animal,      /* Parent class */
            go,          /* Name of function in C++ (must match Python name) */
            n_times      /* Argument(s) */
        );
    }
};

void init_module(py::module m) {
  py::class_<Animal, PyAnimal /* <--- trampoline*/>(m, "Animal")
      .def(py::init<>())
      .def("go_once", &Animal::go_once)
      .def("go", &Animal::go);
}

int main(int, char**) {
  py::scoped_interpreter guard{};

  py::module m("test_module");
  init_module(m);
  py::globals()["m"] = m;

  py::print("[ Eval ]");
  py::exec(R"""(
class Cat(m.Animal):
    def go(self, n_times):
        return 'meow' * n_times

c = Cat()
print(c.go(1))
print(c.go_once())
)""");

  py::print("[ Done ]");

  return 0;
}
