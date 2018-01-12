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
    Container(const string& name)
        : name_(name) {
        print_created(this, name);
    }
    ~Container() {
        print_destroyed(this);
    }

    void add(unique_ptr<UniquePtrHeld> item) {
        items_.push_back(std::move(item));
    }

    void transfer(Container* other) {
        for (auto& item : items_)
            other->add(std::move(item));
        items_.clear();
    }

    string name() const {
        return name_;
    }
private:
    string name_;
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
        .def(py::init<string>())
        .def("add", &Container::add, py::keep_alive<2, 1>())
//        .def("__repr__", &Container::name)
        // Use transitive keep_alive - this container is kept alive by `items_`,
        // but we want `items_` to keep the other container alive, so we do this
        // by keeping *this* container alive.
        .def("transfer", &Container::transfer, py::keep_alive<1, 2>());

    py::dict globals = py::globals();
    globals["m"] = m;

    m.def("sentinel", []() {
       cout << "Sentinel hit" << endl;
    });

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
