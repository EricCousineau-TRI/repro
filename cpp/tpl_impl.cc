#include <iostream>

using namespace std;

template <typename T>
class Test {
public:
  template <typename U>
  void stuff();
};

template <typename T>
template <typename U>
void Test<T>::stuff() {
  cout << T{} << " - " << U{} << endl;
}

int main() {
  Test<int>().stuff<double>();
  return 0;
}
