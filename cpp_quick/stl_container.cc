#include <array>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

template <typename T, size_t Size>
void ensure_size(std::array<T, Size>* values, int size) {
  assert(values->size() == size);
}

template <typename T>
void ensure_size(std::vector<T>* values, int size) {
  values->resize(size);
}

int main() {
  std::array<int, 3> x1;
  std::vector<int> x2;
  ensure_size(&x1, 3);
  ensure_size(&x2, 3);

  cout << "x1: " << x1.size() << endl;
  cout << "x2: " << x2.size() << endl;

  return 0;
}
