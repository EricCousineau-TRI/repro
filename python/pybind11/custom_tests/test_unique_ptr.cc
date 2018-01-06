// Purpose: Debug unique ptr casting.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

void bind_ConstructorStats(py::module &m);

class UniquePtrHeld {
public:
    UniquePtrHeld() = delete;
    UniquePtrHeld(const UniquePtrHeld&) = delete;
    UniquePtrHeld(UniquePtrHeld&&) = delete;

    UniquePtrHeld(int value)
            : value_(value) {
        print_created(this, value);
    }
    ~UniquePtrHeld() {
        print_destroyed(this);
    }
    int value() const { return value_; }
private:
    int value_{};
};

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("_test_unique_ptr");

    bind_ConstructorStats(m);

    py::class_<UniquePtrHeld>(m, "UniquePtrHeld")
            .def(py::init<int>())
            .def("value", &UniquePtrHeld::value);

    m.def("unique_ptr_pass_through",
          [](std::unique_ptr<UniquePtrHeld> obj) {
              return obj;
          });
    m.def("unique_ptr_terminal",
          [](std::unique_ptr<UniquePtrHeld> obj) {
              obj.reset();
          });

    py::dict globals = py::globals();
    globals["m"] = m;

    py::str file;
    if (argc < 2) {
        file = "python/pybind11/custom_tests/test_unique_ptr.py";
    } else {
        file = argv[1];
    }
    py::print(file);
    py::eval_file(file);

    cout << "[ Done ]" << endl;

    return 0;
}
