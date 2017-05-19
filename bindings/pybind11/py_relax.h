#include <cstddef>
#include <cmath>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace detail {

template <typename Method>
struct mem_fn_class {
  // Extract class from a class method pointer.

  // Does std:: have this?
  // Can't find it here
  // http://en.cppreference.com/w/cpp/header/functional
  template <class R, class T, class... Args>
  static T helper(R (T::* pm)(Args...));

  using type = decltype(helper(std::declval<Method>()));
};

}  // namespace detail

// Generic types.

template <typename T>
struct py_relax_type { using type = T; };

/*
 * Filter types in a parameter pack, and marshal them, to call the actual
 * overload.
 * @note Only works for instance methods. No general functions.
 */
template <typename ... Args, typename Method>
auto py_relax_overload(const Method& method) {
  using Base = typename detail::mem_fn_class<Method>::type;
  auto relaxed = [=](
      Base* self, const typename py_relax_type<Args>::type&... args) {
    return (self->*method)(args...);
  };
  return relaxed;
}




// Flexible integer with safety checking
template <typename T = int>
class PyIntegral {
public:
  // Need default ctor to permit type_caster to be constructible per macro
  PyIntegral() = default; 
  PyIntegral(T value)
    : value_(value) {}
  PyIntegral(double value) {
    *this = value;
  }
  operator T() const { return value_; }
  PyIntegral& operator=(T value) {
    value_ = value;
    return *this;
  }
  PyIntegral& operator=(const PyIntegral& other) = default;
  PyIntegral& operator=(const double& value) {
    T tmp = static_cast<T>(value);
    // Ensure they are approximately the same
    double err = value - tmp;
    if (std::abs(err) > 1e-8) {
      std::ostringstream msg;
      msg << "Only integer-valued floating point values accepted"
          << " (input: " << value << ", integral distance: " << err << ")";
      throw std::runtime_error(msg.str());
    }
    return *this = tmp;
  }
private:
  T value_ {};
};

using pyint = PyIntegral<int>;

template <>
struct py_relax_type<int> { using type = pyint; };

namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<PyIntegral<T>> {
  PYBIND11_TYPE_CASTER(pyint, _("pyint"));

  bool load(handle src, bool convert) {
    // Do effective duck-typing.
    type_caster<double> dbl_value;
    type_caster<T> T_value;
    if (dbl_value.load(src, convert)) {
      value = dbl_value;
      return true;
    } else if (T_value.load(src, convert)) {
      value = T_value;
      return true;
    }
    return false;
  }

  static handle cast(pyint src, return_value_policy policy,
                     handle parent) {
    return type_caster<T>::cast(src, policy, parent);
  }
};


} // namespace pybind11
} // namespace detail
