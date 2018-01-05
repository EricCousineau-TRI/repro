// Purpose: Test what avenues might be possible for creating instances in Python
// to then be owned in C++.

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#define PYBIND11_WARN_DANGLING_UNIQUE_PYREF

#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"

namespace py = pybind11;
using namespace py::literals;
using namespace std;

// Returns Python handle to owner.
// CANNOT be used in constructors!!!
template <typename NurseT, typename OwnerT>
py::object expose_ownership(
    const unique_ptr<NurseT>& nurse_ptr, const OwnerT* owner,
    py::object owner_py = {}) {
    if (nurse_ptr) {
        py::print("Has value");
        py::handle nurse_py = py::detail::cast_existing(nurse_ptr.get());
        if (nurse_py) {
            py::print("Has existing value");
            // Expose owner to Python, registering it if needed.
            // This assumes that the lifetime of the owner is appropriately managed!
            if (!owner_py) {
                // TODO(eric.cousineau): Is there a way to ensure that this is not being called in
                // a constructor?
                owner_py = py::cast(owner);
            }
            py::print("Adding: ", nurse_py, owner_py);
            py::print("owner: ", reinterpret_cast<const void*>(owner));
            py::print("- refcount: ", owner_py, owner_py.ref_count());
            py::detail::add_patient(nurse_py.ptr(), owner_py.ptr());
            py::print("- refcount: ", owner_py, owner_py.ref_count());
            owner_py.inc_ref();
            py::print("- refcount: ", owner_py, owner_py.ref_count());
            return owner_py;
        }
        // TODO(eric.cousineau): This is NOT good if we want to always allow the nurse to keep the object alive.
        // (e.g. if it's created within the container, and only a reference is passed).
        // Could just delegate this to the user, have them use `keep_alive<0, 1>`.
    }
    py::print("No ownership");
    // Return empty object.
    return py::object();
}

template <typename NurseT>
void release_ownership(unique_ptr<NurseT> nurse_ptr) {
    if (nurse_ptr && py::detail::cast_existing(nurse_ptr.get())) {
        // Release object to be managed by pybind.
        py::move(std::move(nurse_ptr));
    }
}

enum class KeepAliveType : int {
    Plain = 0,
    KeepAlive,
    ExposeOwnership,
};

template <
    typename T,
    KeepAliveType keep_alive_type>
class Container {
public:
    using Ptr = std::unique_ptr<T>;
    Container(Ptr ptr)
        : ptr_(std::move(ptr)) {
        print_created(this);
    }
    ~Container() {
        if (keep_alive_type == KeepAliveType::ExposeOwnership) {
            release_ownership(std::move(ptr_));
        }
        print_destroyed(this);
    }
    void expose_ownership() {
        if (keep_alive_type != KeepAliveType::ExposeOwnership) {
            throw std::runtime_error("Invalid");
        }
        expose_ownership(ptr_, this);
    }
    T* get() const { return ptr_.get(); }
    Ptr release() { return std::move(ptr_); }

    static void def(py::module &m, const std::string& name) {
        py::class_<Container> cls(m, name.c_str());
        if (keep_alive_type == KeepAliveType::KeepAlive) {
            cls.def(py::init<Ptr>(), py::keep_alive<2, 1>());
        } else {
            cls.def(py::init<Ptr>());
        }
        if (keep_alive_type == KeepAliveType::ExposeOwnership) {
            cls.def("expose_ownership", &Container::expose_ownership);
        }
        cls.def("get", &Container::get);
        cls.def("release", &Container::release);
    }
private:
    Ptr ptr_;
};

void bind_ConstructorStats(py::module &m);

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::module m("test_unique_ptr_keep_alive");

    bind_ConstructorStats(m);

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
    py::class_<UniquePtrHeld>(m, "UniquePtrHeld")
        .def(py::init<int>())
        .def("value", &UniquePtrHeld::value);

    Container<UniquePtrHeld, KeepAliveType::Plain>::def(
        m, "ContainerPlain");
    Container<UniquePtrHeld, KeepAliveType::KeepAlive>::def(
        m, "ContainerKeepAlive");
    Container<UniquePtrHeld, KeepAliveType::ExposeOwnership>::def(
        m, "ContainerExposeOwnership");


    py::dict globals = py::globals();
    globals["m"] = m;
    globals["ConstructorStats"] = m.attr("ConstructorStats");

    py::str file;
    if (argc < 2) {
        file = "python/pybind11/custom_tests/test_unique_ptr_keep_alive.py";
    } else {
        file = argv[1];
    }
    py::print(file);
    py::eval_file(file);

    cout << "[ Done ]" << endl;

    return 0;
}
