#include "python/bindings/pymodule/tpl/simple_converter.h"

#include <cstddef>
#include <cmath>
#include <sstream>
#include <string>

#include "cpp/name_trait.h"

using namespace std;
using namespace simple_converter;

template <typename T = float, typename U = int16_t>
class Base;
NAME_TRAIT_TPL(Base)

// Simple base class.
template <typename T, typename U>
class Base {
 public:
  Base(T t, U u)
    : t_(t),
      u_(u) {}
  template <typename Tc, typename Uc>
  Base(const Base<Tc, Uc>& other)
    : Base(static_cast<T>(other.t_),
           static_cast<U>(other.u_)) {}

  T t() const { return t_; }
  U u() const { return u_; }

  virtual U optional(T value) const {
    cout << py_name() << endl;
    return static_cast<U>(value);
  }

  U dispatch(T value) const {
    cout << "cpp.dispatch [" << py_name() << "]:\n";
    cout << "  ";
    U pv = pure(value);
    cout << "  ";
    U ov = optional(value);
    return pv + ov;
  }

  static string py_name() {
    return "Base__T_" + name_trait<T>::name() +
      "__U_" + name_trait<U>::name();
  }

 private:
  template <typename Tc, typename Uc> friend class Base;
  T t_{};
  U u_{};
};


int main() {
  using PackA = type_pack<float, int>;
  using PackB = type_pack<int, float>;

  SimpleConverter<Base> converter;
  converter.AddCopyConveter<PackA, PackB>();
  converter.AddCopyConveter<PackB, PackA>();

  return 0;
}
