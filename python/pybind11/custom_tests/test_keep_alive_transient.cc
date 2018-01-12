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

class Container {
public:
    Container() {}

    void add(unique_ptr<UniquePtrHeld> item) {
        items_.push_back(std::move(item));
    }

    void transfer(Container* other) {
        for (auto& item : items_)
            other->add(std::move(item));
        items_.clear();
    }
private:
    vector<unique_ptr<UniquePtrHeld>> items_;
};

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("_test_keep_alive_transient");

    bind_ConstructorStats(m);

    py::class_<UniquePtrHeld>(m, "UniquePtrHeld")
            .def(py::init<int>())
            .def("value", &UniquePtrHeld::value);

    py::class_<Container>(m, "Container")
        .def(py::init<>())
        .def("add", &Container::add, py::keep_alive<2, 1>())
        .def("transfer", &Container::transfer, py::keep_alive<2, 1>());

    py::dict globals = py::globals();
    globals["m"] = m;

    py::str file;
    if (argc < 2) {
        file = "python/pybind11/custom_tests/test_keep_alive_transient.py";
    } else {
        file = argv[1];
    }
    py::print(file);
    py::eval_file(file);

    cout << "[ Done ]" << endl;

    return 0;
}
