#include <cassert>
#include <iostream>
#include <functional>

using namespace std;

typedef int (func_t)(int);

int main() {
  // int c = 10;
  std::function<func_t> func = [](int x) {
    // This also causes a segfault???
    int c = 10;
    return c * x;
  };

  func_t* func_ptr = func.target<func_t>();
  assert(func_ptr != nullptr);
  cout << "---" << endl;
  // Causes segfault. Lambda stores state.
  cout << func_ptr(20) << endl;
  cout << "---" << endl;

  return 0;
}
