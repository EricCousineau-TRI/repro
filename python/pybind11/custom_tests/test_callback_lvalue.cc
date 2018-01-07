// Purpose: Debug unique ptr casting.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

void bind_ConstructorStats(py::module &m);

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("_test_callback_lvalue");

    py::dict globals = py::globals();
    globals["m"] = m;

    bind_ConstructorStats(m);

    struct CppCopyable {
        int value{};
    };
    py::class_<CppCopyable> cls(m, "CppCopyable");
    cls.def(py::init<>());
    cls.def_readwrite("value", &CppCopyable::value);

    // Test mutable lvalue references.
     struct Item {
         Item(int value_in) : value(value_in) { print_created(this, value); }
         Item(const Item& other)
                : value(other.value) {
             print_copy_created(this, value);
         }
         ~Item() { print_destroyed(this); }
         int value{};
     };
     py::class_<Item>(m, "Item")
         .def(py::init<int>())
         .def_readwrite("value", &Item::value);

     struct Container {
         Item* new_ptr() { return new Item{100}; }
         Item* get_ptr() { return &item; }
         Item& get_ref() { return item; }

         // Test casting behavior.
         py::object cast_copy() {
             return py::cast(item);
         }
         py::object cast_ptr() { return py::cast(&item); }
         py::object cast_ref() {
             return py::cast(item, py::return_value_policy::reference);
         }

         Item item{10};
     };
     py::class_<Container>(m, "Container")
         .def(py::init<>())
         .def("new_ptr", &Container::new_ptr)
         .def("get_ptr_unsafe", &Container::get_ptr, py::return_value_policy::automatic_reference)
         .def("get_ref_unsafe", &Container::get_ref)
         .def("get_ptr", &Container::get_ptr, py::return_value_policy::reference_internal)
         .def("get_ref", &Container::get_ref, py::return_value_policy::reference_internal)
         .def("cast_copy", &Container::cast_copy)
         .def("cast_ref", &Container::cast_ref);

    {
        py::object c = py::eval("m.Container()");
        c.attr("cast_copy")();
    }

    {
        CppCopyable cpp_obj_orig{1};
        py::object obj = py::cast(cpp_obj_orig);
        obj.attr("value") = 20;
        CppCopyable& cpp_obj = py::cast<CppCopyable&>(obj);
        cpp_obj.value = 200;
        py::print("value: ", cpp_obj_orig.value, cpp_obj.value, obj.attr("value"));

        // Should be: "value:  1 200 200"
    }

    {
        py::dict locals;
        py::exec(R"""(
    def incr(obj):
        obj.value += 1
        )""", py::globals(), locals);

        auto func = py::cast<std::function<void(CppCopyable&)>>(locals["incr"]);
        CppCopyable cpp_obj{10};
        func(cpp_obj);
        py::print("value: ", cpp_obj.value);

        // Without patch: "value: 10"
        // With patch: "value: 11"
    }

    // Works as expected with patch, because pybind will copy the instance when
    // binding.
    py::cpp_function func_ref(
        [](std::function<void(CppCopyable&)> f, int value) {
            CppCopyable obj{value};
            f(obj);
            return obj;
        });
    m.attr("callback_mutate_copyable_cpp_ref") = func_ref;
    // Works as expected, because pybind will not copy the instance.
    py::cpp_function func_ptr(
        [](std::function<void(CppCopyable*)> f, int value) {
            CppCopyable obj{value};
            f(&obj);
            return obj;
        });
    m.attr("callback_mutate_copyable_cpp_ptr") = func_ptr;

    py::str file;
    if (argc < 2) {
        file = "python/pybind11/custom_tests/test_callback_lvalue.py";
    } else {
        file = argv[1];
    }
    py::print(file);
    py::eval_file(file);

    cout << "[ Done ]" << endl;

    return 0;
}
