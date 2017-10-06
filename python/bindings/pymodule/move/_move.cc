// Purpose: Test what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
using namespace std;

namespace move {

class Test {
 public:
  Test(int value)
      : value_(value) {
    cout << "Test::Test(int)\n";
  }
  ~Test() {
    cout << "Test::~Test()\n";
  }

  int value() const { return value_; }

 private:
  int value_{};
};

unique_ptr<Test> check_creation(py::function create_obj) {
  auto PyMove = py::module::import("pymodule.move.py_move").attr("PyMove");
  py::object obj_move = create_obj();
  auto locals = py::dict("obj_move"_a=obj_move, "PyMove"_a=PyMove);
  bool is_good =
      py::eval("isinstance(obj_move, PyMove)", py::globals(), locals).cast<bool>();
  if (!is_good) {
    throw std::runtime_error("Must return a PyMove instance");
  }
  /// using itype = intrinsic_t<type>;
  /// type_caster - operator *itype&()
//  auto ptr = py::cast<shared_ptr<Test>>(obj_move);
  py::object obj = obj_move.attr("release")();
  unique_ptr<Test> in = py::cast<unique_ptr<Test>>(std::move(obj));
  return in;
}

PYBIND11_MODULE(_move, m) {
  m.doc() = "Check move possibilites";

  py::class_<Test>(m, "Test")
    .def(py::init<int>())
    .def("value", &Test::value);

  m.def("check_creation", &check_creation);
}

}  // namespace move

// Export this to get access as we desire.
void custom_init_move(py::module& m) {
  move::PYBIND11_CONCAT(pybind11_init_, _move)(m);
}
