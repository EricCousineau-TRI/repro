// Purpose: Show perfect-forwarding as a way to explicitly avoid copy elision
// without knowing the method return type.

#include <iostream>
#include <utility>

#include "name_trait.h"

using std::cout;
using std::endl;

#define EVAL(x) \
    std::cout << ">>> " #x ";" << std::endl; \
    x; \
    cout << std::endl
#define EVAL_SCOPED(x) \
    std::cout << ">>> scope { " #x " ; }" << std::endl; \
    { \
        x; \
        std::cout << "   <<< [ exiting scope ]" << std::endl; \
    } \
    std::cout << std::endl

template <typename T>
std::string type_name_of(const T&) {
    return name_trait<T>::name();
}

int x{};

int ret_value() { return x; }
int& ret_ref() { return x; }
int&& ret_rref() {
  int y = x;
  return std::move(y);
}
const int& ret_cref() { return x; }

int main() {
  EVAL({ auto x = ret_value(); cout << name_trait<decltype(x)>::name(); });
  EVAL({ auto&& x = ret_value(); cout << name_trait<decltype(x)>::name(); });

  EVAL({ auto x = ret_ref(); cout << name_trait<decltype(x)>::name(); });
  EVAL({ auto&& x = ret_ref(); cout << name_trait<decltype(x)>::name(); });

  EVAL({ auto x = ret_rref(); cout << name_trait<decltype(x)>::name(); });
  EVAL({ auto&& x = ret_rref(); cout << name_trait<decltype(x)>::name(); });

  EVAL({ auto x = ret_cref(); cout << name_trait<decltype(x)>::name(); });
  EVAL({ auto&& x = ret_cref(); cout << name_trait<decltype(x)>::name(); });
}
