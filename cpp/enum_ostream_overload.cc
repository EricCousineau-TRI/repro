#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::ostream;

enum ThatThing {
  ValueA = 0,
  ValueB = 1,
};

string to_string(ThatThing x) {
  switch (x) {
    case ValueA: return "ValueA";
    case ValueB: return "ValueB";
  }
}

ostream& operator<<(ostream& os, ThatThing x) {
  return os << "ThatThing{" << to_string(x) << "}";
}

int main() {
  cout << ValueA << endl;
  cout << to_string(ValueA) << endl;
  cout << to_string(ThatThing::ValueA) << endl;
  cout << ValueB << endl;
  cout << to_string(ValueB) << endl;
  cout << 10 << endl;
  cout << to_string(10) << endl;

  return 0;
}

/*
Output:

ThatThing{ValueA}
ValueA
ThatThing{ValueB}
ValueB
10
10
*/
