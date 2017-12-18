// Purpose: Base what avenues might be possible for creating instances in Python
// to then be owned in C++.

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

class SimpleType {
 public:
    SimpleType(int value)
        : value_(value) {
      cout << "SimpleType::SimpleType()" << endl;
    }
    ~SimpleType() {
      cout << "SimpleType::~SimpleType()" << endl;
    }
    int value() const { return value_; }
 private:
    int value_{};
};

// Check casting.
py::handle check(py::handle py_in) {
  cout << "Pass through: " << py_in.ref_count() << endl;
  if (py_in.ref_count() == 1) {
    // Steal as an object.
    auto obj = py::reinterpret_steal<py::object>(py_in);
    cout << "- Stole: " << obj.ref_count() << endl;
    // Causes object to be destructed here.
    return obj.release();
  } else {
    return py_in;
  }
//  cout << "- Return" << endl;
}

PYBIND11_MODULE(_move, m) {
  // Make sure this also still works with non-virtual, non-wrapper types.
  py::class_<SimpleType>(m, "SimpleType")
      .def(py::init<int>())
      .def("value", &SimpleType::value);
  m.def("check", &check);
}

// Export this to get access as we desire.
void custom_init_move(py::module& m) {
  PYBIND11_CONCAT(pybind11_init_, _move)(m);
}

void check_ref_count() {
  cout << "\n[ check_ref_count ]\n";
  py::exec(R"(
obj = move.SimpleType(256)
move.check(obj)
move.check(move.SimpleType(12))
del obj
)");
}

int main() {
  {
    py::scoped_interpreter guard{};

    py::module m("_move");
    custom_init_move(m);
    py::globals()["move"] = m;

    check_ref_count();
  }

  cout << "[ Done ]" << endl;

  return 0;
}
