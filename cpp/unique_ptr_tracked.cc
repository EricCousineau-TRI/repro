#include <cassert>

#include <iostream>
#include <memory>

using namespace std;

template <typename T>
class Tracker {
 public:
  Tracker() = default;
  Tracker(std::default_delete<T>&&) {
    cout << "Converted from original" << endl;
  }
  Tracker(Tracker&& other) {
    cout << "Moved Tracker" << endl;
  }
  ~Tracker() {
    cout << "Deleted Tracker" << endl;
  }
  Tracker& operator=(std::default_delete<T>&&) {
    cout << "Converted from original" << endl;
    return *this;
  }
  void operator()(T* ptr) {
    // Only called when `ptr` is non-null.
    cout << "Delete: " << ptr << endl;
    assert(ptr);
    delete ptr;
  }
  operator std::default_delete<T>() {
    cout << "Converted to original" << endl;
    return std::default_delete<T>{};
  }
};

struct Test { int value{}; };

int main() {
  unique_ptr<Test> ptr = make_unique<Test>();
  unique_ptr<Test, Tracker<Test>> ptr_2 = std::move(ptr);
  ptr = std::move(ptr_2);
  ptr_2 = std::move(ptr);
  cout << "Done" << endl;
  return 0;
}
