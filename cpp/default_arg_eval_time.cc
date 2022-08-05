#include <iostream>

using namespace std;

int ProducesValue() {
  static int value = 0;
  return value++;
}

int TakesValue(int value = ProducesValue()) {
  return value;
}

int main() {
  cout
      << TakesValue() << endl
      << TakesValue() << endl
      << TakesValue(10) << endl
      << TakesValue() << endl;

  return 0;
}

/*
0
1
10
2
*/
