#include <iostream>
#include <typeinfo>

#include "cpp/type_pack.h"

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

  Pack::visit_if<is_different_from<int>>(visitor_test{5, 15.});

  cout << typeid(Pack::type<0>).name() << endl;
  cout << typeid(Pack::type<1>).name() << endl;

  return 0;
}
