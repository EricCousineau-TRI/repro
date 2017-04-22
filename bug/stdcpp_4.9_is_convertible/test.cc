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
struct A {};
struct B {};
int main() {
  bool bad = std::is_convertible<
        std::unique_ptr<A>, std::shared_ptr<B>>::value;
  std::cout << "Bad: " << bad << std::endl;
  return bad ? 1 : 0;
}
