#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

namespace inherit_check {

/* Custom pybind11 overload stuff */

#define EX_PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...) { \
        pybind11::gil_scoped_acquire gil; \
        if (dynamic_cast<const cname *>(this) == nullptr) { \
          throw std::runtime_error("Invalid cast"); \
        } \
        pybind11::function overload = pybind11::get_overload(dynamic_cast<const cname *>(this), name); \
        if (overload) { \
            auto o = overload(__VA_ARGS__); \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) { \
                static pybind11::detail::overload_caster_t<ret_type> caster; \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
            } \
            else return pybind11::detail::cast_safe<ret_type>(std::move(o)); \
        } \
    }

#define EX_PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...) \
    EX_PYBIND11_OVERLOAD_INT(ret_type, cname, name, __VA_ARGS__) \
    std::cout << "pybind11: using defualt" << std::endl; \
    return cname::fn(__VA_ARGS__)

#define EX_PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...) \
    EX_PYBIND11_OVERLOAD_INT(ret_type, cname, name, __VA_ARGS__) \
    pybind11::pybind11_fail("Tried to call pure virtual function \"" #cname "::" name "\"");

#define EX_PYBIND11_OVERLOAD(ret_type, cname, fn, ...) \
    EX_PYBIND11_OVERLOAD_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

#define EX_PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...) \
    EX_PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, #fn, fn, __VA_ARGS__)

/* */

// Simple base class.
class Base {
 public:
  virtual int pure(int value) { return 0; }
  virtual int optional(int value) {
    return 0;
  }
  int dispatch(int value) {
    cout << "cpp.dispatch: " << pure(value) << " " << optional(value) << endl;
    return pure(value) + optional(value);
  }
};

class PyBase : public Base {
 public:
  int pure(int value) override {
    EX_PYBIND11_OVERLOAD(int, Base, pure, value);
  }
  int optional(int value) override {
    EX_PYBIND11_OVERLOAD(int, Base, optional, value);
  }
};

class CppExtend : public Base {
 public:
  int pure(int value) override {
    cout << "cpp.pure=" << value << endl;
    return value;
  }
  int optional(int value) override {
    cout << "cpp.optional=" << value << endl;
    return 10 * value;
  }
};

PYBIND11_PLUGIN(_inherit_check) {
  py::module m("_inherit_check", "Simple check on inheritance");

  py::class_<Base, PyBase> base(m, "Base");
  base
    .def(py::init<>())
    .def("pure", &Base::pure)
    .def("optional", &Base::optional)
    .def("dispatch", &Base::dispatch);

  py::class_<CppExtend>(m, "CppExtend", base)
    .def(py::init<>());

  return m.ptr();
}

}  // namespace inherit_check
