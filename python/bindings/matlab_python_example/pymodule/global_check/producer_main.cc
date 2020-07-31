#include <iostream>

#include "producer.h"

using namespace std;
using namespace global_check;

ostream& operator<<(ostream& os, const pair<string, double>& value) {
  os << value.first << " - " << value.second;
  return os;
}

int main() {
  const double value = 2;
  auto p1 = Producer(value);
  cout << p1 << endl;
  auto p2 = ProducerB(value);
  cout << p2 << endl;
  return 0;
}
