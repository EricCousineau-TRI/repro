#include <iostream>
#include <typeindex>
#include <typeinfo>

using std::cout;
using std::endl;

#define assert_ex(expr) \
    if (!(expr)) { \
      throw std::runtime_error("assert failed: " #expr); \
    }

class ptr_erased {
 public:
  template <typename T>
  ptr_erased(T* ptr)
    : ptr_(ptr), const_(false), tinfo_(typeid(T).hash_code()) {}

  template <typename T>
  ptr_erased(const T* ptr)
    : ptr_(const_cast<T*>(ptr)), const_(true),
      tinfo_(typeid(T).hash_code()) {}

  ptr_erased() = default;

  operator bool() const {
    return ptr_ != nullptr;
  }

  template <typename T>
  const T* cast() const {
    assert_ex(typeid(T).hash_code() == tinfo_);
    return reinterpret_cast<const T*>(ptr_);
  }

  template <typename T>
  T* mutable_cast() const {
    assert_ex(!const_);
    assert_ex(typeid(T).hash_code() == tinfo_);
    return reinterpret_cast<T*>(ptr_);
  }
 
 private:
  void* ptr_{};
  bool const_{};
  size_t tinfo_{};
};

int main() {
  int x{};
  const double y{};
  ptr_erased ptr = &x;
  assert_ex(ptr);
  cout << *ptr.cast<int>() << endl;
  *ptr.mutable_cast<int>() = 10;
  cout << x << endl;
  ptr = &y;
  cout << *ptr.cast<double>() << endl;
  try {
    *ptr.mutable_cast<double>() = 1.5;  // Should trigger exception.
  } catch (const std::runtime_error& e) {
    cout << e.what() << endl;
  }

  return 0;
}
