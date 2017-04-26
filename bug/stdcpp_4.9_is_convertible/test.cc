/**
commit 81fe3855414073162869a8329dd0d5c6f7ed35a0
Author: Brad King <brad.king@kitware.com>
Date:   Wed Feb 24 14:43:35 2016 -0500

Optimization: Fix addCost unique_ptr support on libstdc++ 4.9 and below

The libstdc++ 4.9 implementation rejects our `addCost` overload because
it incorrectly considers `unique_ptr<A>` to be convertible to
`shared_ptr<B>` for unrelated types `A` and `B` as seen by this example:

```c++
    #include <type_traits>
    #include <memory>
    struct A {};
    struct B {};
    int main() {
      return std::is_convertible<std::unique_ptr<A>, std::shared_ptr<B>>::value? 0 : 1;
    }
```

Work around this problem by providing an explicit overload for
`unique_ptr` costs.
**/

#include <type_traits>
#include <memory>
#include <iostream>

using std::cout;
using std::endl;

struct A {};
struct B {};

template<typename A, typename B>
struct workaround
    : std::is_convertible<A, B> { };
template<typename A, typename B>
struct workaround<std::unique_ptr<A>, std::shared_ptr<B>>
    : std::is_convertible<A*, B*> { };

template<typename T1, typename T2>
void check() {
  cout
    << "is_convertible: " << std::is_convertible<T1, T2>::value << endl
    << "workaround: " << workaround<T1, T2>::value << endl
    << endl;
}

int main() {
  cout
      // https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
#ifdef __clang__
       << "__clang__: " << __clang__ << endl
#endif // #ifdef __clang__
       << "__GNUC__: " << __GNUC__ << endl
       << "__GNUC_MINOR__: " << __GNUC_MINOR__ << endl
       << "__GNUC_PATCHLEVEL__: " << __GNUC__ << endl
      // https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_macros.html
      // https://patchwork.ozlabs.org/patch/716321/
       << "__GLIBCXX__: " << __GLIBCXX__ << endl;
      // There is only a recent patch to include a richer version numbering 
      // for libstdc++ (20170117)
      // https://gcc.gnu.org/ml/libstdc++/2017-01/txtQphYvbjBvG.txt
      // https://patchwork.ozlabs.org/patch/716321/
       // << "_GLIBCXX_RELEASE: " << _GLIBCXX_RELEASE << endl;

  using APtr = A*;
  using AUPtr = std::unique_ptr<A>;
  using ASPtr = std::shared_ptr<A>;
  using BPtr = B*;
  using BUPtr = std::unique_ptr<B>;
  using BSPtr = std::shared_ptr<B>;

  check<APtr, BPtr>();
  check<APtr, BUPtr>();
  check<APtr, BSPtr>();
  cout << endl;

  check<AUPtr, BPtr>();
  check<AUPtr, BUPtr>();
  check<AUPtr, BSPtr>(); // Error
  cout << endl;

  check<ASPtr, BPtr>();
  check<ASPtr, BUPtr>();
  check<ASPtr, BSPtr>();
  cout << endl;

  return 0;
}
