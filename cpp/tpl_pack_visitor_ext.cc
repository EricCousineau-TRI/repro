#include <iostream>
#include <typeinfo>
#include <utility>

// - BEGIN: Added
template <typename ... Ts>
struct type_pack {
  template <template <typename...> class Tpl>
  using bind = Tpl<Ts...>;

  template <typename Visitor>
  static void visit(Visitor&& visitor) {
    int dummy[] = {(std::forward<Visitor>(visitor).template run<Ts>(), 0)...};
  }
};

using namespace std;

struct visitor_test {
  int arg_1;
  double arg_2;

  template <typename T>
  void run() {
    cout << "T: " << typeid(T).name() << endl;
    cout << arg_1 << " - " << arg_2 << endl;
  }
};

int main() {
  using Pack = type_pack<double, int>;
  Pack::visit(visitor_test{1, 2.0});

  return 0;
}
