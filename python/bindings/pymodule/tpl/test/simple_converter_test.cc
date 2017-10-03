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
  explicit Base(const Base<Tc, Uc>& other)
    : Base(static_cast<T>(other.t_),
           static_cast<U>(other.u_)) {
    cout << "Copy: " << py_name() << " <- " << Base<Tc, Uc>::py_name() << "\n";
  }

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

  using BaseA = Base<float, int>;
  using BaseB = Base<int, float>;

  BaseA a(1.5, 10);
  BaseB b(1, 10.5);

  std::unique_ptr<BaseA> a_b = converter.Convert<BaseA>(b);
  cout << b.t() << " " << b.u() << endl;
  cout << a_b->t() << " " << a_b->u() << endl;

  cout << static_cast<float>(1) << endl;

  // converter.Convert<BaseB>(b);  // Fails as expected.
  // converter.Convert<Base<double, uint>>(a);  // Fails as expected.

  return 0;
}
