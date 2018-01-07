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
    py::module m("_test_callback_lvalue");

//    bind_ConstructorStats(m);

    struct CppCopyable {
        int value{};
    };
    py::class_<CppCopyable> cls(m, "CppCopyable");
    cls.def(py::init<>());
    cls.def_readwrite("value", &CppCopyable::value);

    CppCopyable cpp_obj_orig{1};
    py::object obj = py::cast<CppCopyable&>(cpp_obj_orig);
//    py::object obj = cls();
    obj.attr("value") = 20;
    CppCopyable& cpp_obj = py::cast<CppCopyable&>(obj);
    cpp_obj.value = 200;
    py::print("value: ", cpp_obj_orig.value, cpp_obj.value, obj.attr("value"));

    // Output:
    // value:  1 200 200

//    // Does not work as expected, because pybind will copy the instance when
//    // binding.
//    py::cpp_function func_ref(
//        [](std::function<void(CppCopyable&)> f, int value) {
//            CppCopyable obj{value};
//            f(obj);
//            return obj;
//        });
//    m.attr("callback_mutate_copyable_cpp_ref") = func_ref;
//    // Works as expected, because pybind will not copy the instance.
//    py::cpp_function func_ptr(
//        [](std::function<void(CppCopyable*)> f, int value) {
//            CppCopyable obj{value};
//            f(&obj);
//            return obj;
//        });
//    m.attr("callback_mutate_copyable_cpp_ptr") = func_ptr;

//    py::dict globals = py::globals();
//    globals["m"] = m;
//
//    py::str file;
//    if (argc < 2) {
//        file = "python/pybind11/custom_tests/test_callback_lvalue.py";
//    } else {
//        file = argv[1];
//    }
//    py::print(file);
//    py::eval_file(file);

    cout << "[ Done ]" << endl;

    return 0;
}
