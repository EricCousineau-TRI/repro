#include <iostream>
#include <memory>

using namespace std;

template <typename T>
struct cloneable_helper {
  template <typename U = T>
  static std::true_type Check(
      decltype(std::declval<U>().Clone())* ptr);
  static std::false_type Check(...);

  template <typename U = T>
  static auto Do(const U& other) { return other.Clone(); }
};

template <typename T>
using is_cloneable = decltype(cloneable_helper<T>::Check(nullptr));

template <typename T, typename Token = void>
struct copyable_helper {
  template <typename U = T>
  static std::true_type Check(
      decltype(U(std::declval<const U&>()))* ptr);
  static std::false_type Check(...);

  // Should define some sort of emplace constructor if needed on the stack.
  template <typename U = T>
  static U* Do(const U& other) { return new U(other); }
};

template <typename T>
using is_copyable = decltype(copyable_helper<T>::Check(nullptr));

struct CloneablePrivate {
 private:
  CloneablePrivate(const CloneablePrivate&) = default;
  unique_ptr<CloneablePrivate> Clone() const {
    return unique_ptr<CloneablePrivate>(new CloneablePrivate(*this));
  }

  template <typename T>
  friend struct cloneable_helper;
};

struct NonCloneable {};

struct CopyablePrivate {
 private:
  CopyablePrivate(const CopyablePrivate&) = default;

  friend struct copyable_helper<CopyablePrivate, int>;
};

struct NonCopyable {
 private:
  NonCopyable(const NonCopyable&) = default;
};

int main() {
  cout << is_cloneable<CloneablePrivate>::value << endl;
  auto clone = cloneable_helper<CloneablePrivate>::Do(CloneablePrivate{});
  cout << clone.get() << endl;
  cout << is_cloneable<NonCloneable>::value << endl;

  cout << is_copyable<CopyablePrivate>::value << endl;
  // unique_ptr<CopyablePrivate> copy(
  //     copyable_helper<CopyablePrivate>::Do(CopyablePrivate{}));
  // cout << copy.get() << endl;
  cout << is_copyable<NonCopyable>::value << endl;

  return 0;
}

/*
Output:

1
0x802c30
0
1
0x7ffdf4d42800
0

*/
