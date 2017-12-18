#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace std;

int main(int, char**) {
  py::scoped_interpreter guard{};

  cout << "Start" << endl;

  // test_inheritance
  class Pet {
  public:
      Pet(const std::string &name, const std::string &species)
          : m_name(name), m_species(species) {}
      std::string name() const { return m_name; }
      std::string species() const { return m_species; }
  private:
      std::string m_name;
      std::string m_species;
  };

  class Dog : public Pet {
  public:
      Dog(const std::string &name) : Pet(name, "dog") {}
      std::string bark() const { return "Woof!"; }
  };

  py::module m("test_poly");
  py::class_<Pet> pet_class(m, "Pet");
  pet_class
      .def(py::init<std::string, std::string>())
      .def("name", &Pet::name)
      .def("species", &Pet::species);

  /* One way of declaring a subclass relationship: reference parent's class_ object */
  py::class_<Dog>(m, "Dog", pet_class)
      .def(py::init<std::string>());

  cout << "Registered" << endl;

  py::dict globals = py::globals();
  globals["m"] = m;

  cout << "Eval" << endl;

  py::exec(R"""(
import gc

class PyDog(m.Dog):
    pass

for cls in m.Dog, PyDog:
  molly = [cls("Molly") for _ in range(1)]
  del molly
  gc.collect()
)""");

  cout << "Done" << endl;

  return 0;
}
