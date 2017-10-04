#include <iostream>
#include <memory>

using namespace std;

template <typename T>
struct no_delete {
  constexpr void operator()(T* p) const {}
};

int main() {
  // Have to always specify deleter. Cannot override it at run-time...
  int* raw = new int(10);
  {
    unique_ptr<int, no_delete<int>> ptr(raw);
    // // Does not work
    // unique_ptr<int> ptr_2 = std::move(ptr);
  }
  delete raw;

  // Test allocating with global new()
  raw = static_cast<int*>(::operator new(sizeof(int)));
  delete raw;

  cout << "Done" << endl;
  return 0;
}
