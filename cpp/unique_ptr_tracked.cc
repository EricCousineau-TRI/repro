#include <cassert>

#include <iostream>
#include <memory>

using namespace std;

template <typename T>
class ptr_tracker {
 public:
  static void on_claim(T* ptr) {
    cout << "Claim: " << ptr << endl;
  }

  static void on_release(T* ptr) {
    cout << "Release: " << ptr << endl;
  }

  static void on_delete(T* ptr) {
    cout << "Delete: " << ptr << endl;
    delete ptr;
  }
};

// Option A: Deleter.
// Can't track ownership transfer, though.
template <typename T, typename Tracker = ptr_tracker<T>>
class delete_tracker {
 public:
  delete_tracker() = default;
  // Allow interfacting with trivial deleters.
  template <typename Deleter>
  delete_tracker(Deleter&&) {}
  template <typename Deleter>
  delete_tracker& operator=(Deleter&&) {
    return *this;
  }
  template <typename Deleter>
  operator Deleter() {
    return Deleter{};
  }
  // Only called when `ptr` is non-null.
  void operator()(T* ptr) {
    Tracker::on_delete(ptr);
  }
};

template <typename T>
using unique_ptr_tracked_del = unique_ptr<T, delete_tracker<T>>;


// Option B: Wrap unique_ptr.
template <
  typename T, typename D = std::default_delete<T>,
  typename Tracker = ptr_tracker<T>>
class unique_ptr_tracked {
 public:
  unique_ptr_tracked() = default;
  unique_ptr_tracked(T* ptr)
    : ptr_(ptr) {
    Tracker::on_claim(ptr_.get());
  }
  unique_ptr_tracked(const unique_ptr_tracked&) = delete;
  ~unique_ptr_tracked() {
    Tracker::on_delete(ptr_.release());
  }

  template <typename U, typename E>
  unique_ptr_tracked(std::unique_ptr<U, E>&& ptr) {
    ptr_ = std::move(ptr);
    Tracker::on_claim(ptr_.get());
  }

  template <typename U, typename E, typename S>
  unique_ptr_tracked(unique_ptr_tracked<U, E, S>&& other) {
    ptr_ = std::move(other);  // Let it invoke it's `on_release()`.
    Tracker::on_claim(ptr_.get());
  }

  template <typename U, typename E>
  unique_ptr_tracked& operator=(std::unique_ptr<U, E>&& ptr) {
    ptr_ = std::move(ptr);
    Tracker::on_claim(ptr_.get());
    return *this;
  }
  unique_ptr_tracked& operator=(const unique_ptr_tracked&) = delete;

  template <typename U, typename E>
  operator std::unique_ptr<U, E>() {
    Tracker::on_release(ptr_.get());
    return std::move(ptr_);
  }

  T* get() const { return ptr_.get(); }
  T* release() {
    Tracker::on_release(ptr_.get());
    return ptr_.release();
  }
  void reset(T* ptr) {
    Tracker::on_delete(ptr_.release());
    ptr_.reset(ptr);
    Tracker::on_claim(ptr_.get());
  }
  template <typename U, typename E>
  void swap(std::unique_ptr<U, E>& ptr) {
    Tracker::on_release(ptr_.get());
    ptr_.swap(ptr);
    Tracker::on_claim(ptr_.get());
  }
  void swap(unique_ptr_tracked& other) {
    // Both managed by the same tracker. No problem.
    ptr_.swap(other.ptr_);
  }
  template <typename U, typename E, typename S>
  void swap(unique_ptr_tracked<U, E, S>& other) {
    // N.B. There will be an instance where both `ptr_` will be 'unclaimed'.
    // That's better than both being claimed.
    Tracker::on_release(ptr_.get());
    other.swap(ptr_);
    Tracker::on_claim(ptr_.get());
  }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_.operator->(); }
 private:
  std::unique_ptr<T, D> ptr_;
};

struct Test { int value{}; };

int main() {
  {
    cout << "unique_ptr_tracked" << endl;
    unique_ptr<Test> ptr = make_unique<Test>();
    unique_ptr_tracked<Test> ptr_2 = std::move(ptr);
    unique_ptr_tracked<Test> ptr_3 = std::move(ptr_2);
    ptr = std::move(ptr_3);
    ptr_2 = std::move(ptr);
    assert(ptr_2->value == 0);
    assert(*ptr_2 == *ptr_2.get());
    ptr_3.reset(ptr_2.release());
    ptr_2.swap(ptr_3);
  }
  cout << "---" << endl;
  {
    cout << "unique_ptr_tracked_del" << endl;
    unique_ptr<Test> ptr = make_unique<Test>();
    unique_ptr_tracked_del<Test> ptr_2 = std::move(ptr);
    ptr = std::move(ptr_2);
    unique_ptr_tracked_del<Test> ptr_3 = std::move(ptr);
  }

  return 0;
}

/*
Output:

unique_ptr_tracked
Claim: 0x1c13c30
Release: 0x1c13c30
Claim: 0x1c13c30
Release: 0x1c13c30
Claim: 0x1c13c30
Release: 0x1c13c30
Delete: 0
Claim: 0x1c13c30
Delete: 0
Delete: 0x1c13c30
---
unique_ptr_tracked_del
Delete: 0x1c13c30

*/
