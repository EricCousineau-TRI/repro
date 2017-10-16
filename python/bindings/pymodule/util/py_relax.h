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
 * overload. Like "py::overload_cast"
 * @note Only works for instance methods. No general functions.
 */
template <typename ... Args, typename Method>
auto py_relax_overload_cast(const Method& method) {
  using Base = typename detail::mem_fn_class<Method>::type;
  auto relaxed = [=](
      Base* self, const typename py_relax_type<Args>::type&... args) {
    return (self->*method)(args...);
  };
  return relaxed;
}

template <typename ... Args>
auto py_relax_init() {
  return py::init<typename py_relax_type<Args>::type...>();
}


namespace pybind11 {
namespace detail {

template <typename Type, typename Option, typename... Others>
bool duck_type_cast(Type* pvalue, handle src, bool convert) {
  // Scope this such that it does not accumulate casters for each type
  // on failure.
  bool result{};
  {
    type_caster<Option> opt_value;
    result = opt_value.load(src, convert);
    if (result) {
      // Store value from caster.
      *pvalue = opt_value;
      return true;
    }
  }
  // Delegate to other types
  return duck_type_cast<Type, Others...>(pvalue, src, convert);
}

template <typename Type>
bool duck_type_cast(...) {
  // No successful casts.
  return false;
}

// Duck-type type_caster mixin
template <typename Type, typename... Options>
struct duck_type_caster_mixin {
  bool load_impl(Type* pvalue, handle src, bool convert) {
    return duck_type_cast<Type, Options...>(pvalue, src, convert);
  }

  static handle cast(Type src, return_value_policy policy, handle parent) {
    return type_caster<Type>::cast(src, policy, parent);
  }
};

}  // namespace detail
}  // namespace pybind11


// Using pybind's type_descr constexpr magic.
using py::detail::_;

template <typename T>
struct name_trait {
  static constexpr auto name = _("T");
};
template <>
struct name_trait<int> {
  static constexpr auto name = _("int");
};


// Flexible integer with safety checking

template <typename T = int>
class RelaxIntegral {
public:
  // Need default ctor to permit type_caster to be constructible per macro
  RelaxIntegral() = default; 
  RelaxIntegral(T value)
    : value_(value) {}
  RelaxIntegral(double value) {
    *this = value;
  }
  operator T() const { return value_; }
  RelaxIntegral& operator=(T value) {
    value_ = value;
    return *this;
  }
  RelaxIntegral& operator=(const RelaxIntegral& other) = default;
  RelaxIntegral& operator=(const double& value) {
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

template <typename T>
struct name_trait<RelaxIntegral<T>> {
  static constexpr auto name = name_trait<T>::name + _("_relax");
};

using int_relax = RelaxIntegral<int>;

template <>
struct py_relax_type<int> { using type = int_relax; };

namespace pybind11 {
namespace detail {

// Register type_caster for RelaxIntegral.
template <typename T>
struct type_caster<RelaxIntegral<T>>
    : public duck_type_caster_mixin<RelaxIntegral<T>, T, double> {

  PYBIND11_TYPE_CASTER(RelaxIntegral<T>, name_trait<RelaxIntegral<T>>::name);

  bool load(handle src, bool convert) {
    return this->load_impl(&value, src, convert);
  }
};

}  // namespace detail
}  // namespace pybind11


// Flexible scalar-matrix registration.
// TODO: Could generalize if need be.

template <typename T = Eigen::MatrixXd>
class RelaxMatrix {
public:
  // TODO: Generalize dimensions and all that stuff.
  using Scalar = typename T::Scalar;

  // Need default ctor to permit type_caster to be constructible per macro
  RelaxMatrix() = default; 
  RelaxMatrix(const T& value)
    : value_(value) {}
  RelaxMatrix(const Scalar& value) {
    // Delegate to assignment overload.
    *this = value;
  }

  operator T&() { return value_; }
  operator const T&() const { return value_; }

  RelaxMatrix& operator=(const RelaxMatrix& other) = default;
  RelaxMatrix& operator=(const T& value) {
    value_ = value;
    return *this;
  }
  RelaxMatrix& operator=(const Scalar& value) {
    // Accept scalar.
    value_.resize(1, 1);
    value_(0) = value;
    return *this;
  }
private:
  T value_ {};
};

template <typename T>
struct name_trait<RelaxMatrix<T>> {
  // TODO: Develop better (unique) naming for this.
  // See //cpp/eigen/matrix_hstack_xpr_tpl stuff.
  static constexpr auto name = name_trait<T>::name + _("_relax_matrix");
};

// TODO: Use `is_eigen_densebase` from pybind11 to make this more expansive.
template <>
struct py_relax_type<Eigen::MatrixXd> {
  using type = RelaxMatrix<Eigen::MatrixXd>;
};

namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<RelaxMatrix<T>>
    : public duck_type_caster_mixin<RelaxMatrix<T>, T, double> {

  PYBIND11_TYPE_CASTER(RelaxMatrix<T>, name_trait<RelaxMatrix<T>>::name);

  bool load(handle src, bool convert) {
    return this->load_impl(&value, src, convert);
  }
};

} // namespace pybind11
} // namespace detail
