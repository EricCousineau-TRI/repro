#include <iostream>
#include <typeindex>
#include <typeinfo>

using std::cout;
using std::endl;

#define assert_ex(expr) \
    if (!(expr)) { \
      throw std::runtime_error("assert failed: " #expr); \
    }

struct type_hash_ptr {
  using type = const std::type_info*;
  template <typename T>
  static type hash() { return &typeid(T); }
};

class ptr_erased {
 public:
  using type_hasher = type_hash_ptr;

  ptr_erased() = default;
  ptr_erased(std::nullptr_t) : ptr_erased() {}

  template <typename T>
  ptr_erased(T* ptr)
    : ptr_(ptr), is_const_(false), type_(type_hasher::hash<T>()) {}

  template <typename T>
  ptr_erased(const T* ptr)
    : ptr_(const_cast<T*>(ptr)), is_const_(true),
      type_(type_hasher::hash<T>()) {}

  operator bool() const {
    return ptr_ != nullptr;
  }

  bool is_const() const { return is_const_; }

  type_hasher::type type_hash() const { return type_; }

  template <typename T>
  const T* cast() const {
    assert_ex(type_hasher::hash<T>() == type_);
    return reinterpret_cast<const T*>(ptr_);
  }

  template <typename T>
  T* mutable_cast() const {
    assert_ex(!is_const_);
    assert_ex(type_hasher::hash<T>() == type_);
    return reinterpret_cast<T*>(ptr_);
  }
 
 private:
  void* ptr_{};
  bool is_const_{};
  type_hasher::type type_{};
};

int main() {
  int x{1};
  const double y{1.5};
  ptr_erased ptr = &x;
  assert_ex(ptr);
  cout << *ptr.cast<int>() << endl;
  *ptr.mutable_cast<int>() = 10;
  cout << x << endl;
  ptr = &y;
  cout << *ptr.cast<double>() << endl;
  try {
    cout << *ptr.cast<int>() << endl;  // Should trigger exception.
  } catch (const std::runtime_error& e) {
    cout << e.what() << endl;
  }
  try {
    *ptr.mutable_cast<double>() = 1.5;  // Should trigger exception.
  } catch (const std::runtime_error& e) {
    cout << e.what() << endl;
  }
  ptr = nullptr;

  return 0;
}
