#include <cassert>

#include <iostream>
#include <memory>

using namespace std;

// Allow tracking a unique ptr.
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
using unique_ptr_tracked = unique_ptr<T, Tracker<T>>;

struct Test { int value{}; };

int main() {
  unique_ptr<Test> ptr = make_unique<Test>();
  unique_ptr_tracked<Test> ptr_2 = std::move(ptr);
  unique_ptr_tracked<Test> ptr_3 = std::move(ptr_2);
  ptr = std::move(ptr_3);
  ptr_2 = std::move(ptr);
  cout << "Done" << endl;
  return 0;
}
