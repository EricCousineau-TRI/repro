#include <cassert>

#include <iostream>
#include <memory>

using namespace std;

// Allow tracking a unique ptr.
// ... er... Can this track the value too?
// `Deleter` must be trivially destructible.
template <typename T>
class Tracker {
 public:
  Tracker() = default;
  template <typename Deleter>
  Tracker(Deleter&&) {
    cout << "Moved from another unique_ptr" << endl;
  }
  ~Tracker() {
    cout << "Tracker removed" << endl;
  }
  template <typename Deleter>
  Tracker& operator=(Deleter&&) {
    cout << "Moved from another unique_ptr" << endl;
    return *this;
  }
  void operator()(T* ptr) {
    // Only called when `ptr` is non-null.
    cout << "Ptr Delete: " << ptr << endl;
    assert(ptr);
    delete ptr;
  }
  template <typename Deleter>
  operator Deleter() {
    cout << "Moved to another unique_ptr" << endl;
    return Deleter{};
  }
};

template <typename T>
using unique_ptr_tracked_meh_meh = unique_ptr<T, Tracker<T>>;

template <typename T, typename D = std::default_delete<T>>
class unique_ptr_tracked {
 public:
  unique_ptr_tracked() = default;
  unique_ptr_tracked(T* ptr)
    : ptr_(ptr) {
    on_claim();
  }
  unique_ptr_tracked(const unique_ptr_tracked&) = delete;
  ~unique_ptr_tracked() {
    on_delete();
  }

  template <typename U, typename E>
  unique_ptr_tracked(std::unique_ptr<U, E>&& ptr) {
    ptr_ = std::move(ptr);
    on_claim();
  }

  template <typename U, typename E>
  unique_ptr_tracked(unique_ptr_tracked<U, E>&& other) {
    ptr_ = std::move(other);  // Let it invoke it's `on_release()`.
    on_claim();
  }

  template <typename U, typename E>
  unique_ptr_tracked& operator=(std::unique_ptr<U, E>&& ptr) {
    ptr_ = std::move(ptr);
    on_claim();
    return *this;
  }
  unique_ptr_tracked& operator=(const unique_ptr_tracked&) = delete;

  template <typename U, typename E>
  operator std::unique_ptr<U, E>() {
    on_release();
    return std::move(ptr_);
  }

  T* get() const { return ptr_.get(); }
  T* release() {
    on_release();
    return ptr_.release();
  }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_.operator->(); }
 private:
  std::unique_ptr<T, D> ptr_;

  void on_claim() {
    cout << "Claim: " << ptr_.get() << endl;
  }

  void on_release() {
    cout << "Release: " << ptr_.get() << endl;
  }

  void on_delete() {
    cout << "Delete: " << ptr_.get() << endl;
  }

  // template <typename U, typename E>
  // friend class unique_ptr_tracked;
};



struct Test { int value{}; };

int main() {
  unique_ptr<Test> ptr = make_unique<Test>();
  // unique_ptr_tracked_meh<Test> ptr_2 = std::move(ptr);
  // unique_ptr_tracked_meh<Test> ptr_3 = std::move(ptr_2);
  unique_ptr_tracked<Test> ptr_2 = std::move(ptr);
  unique_ptr_tracked<Test> ptr_3 = std::move(ptr_2);
  ptr = std::move(ptr_3);
  ptr_2 = std::move(ptr);
  cout << "Done" << endl;
  return 0;
}
