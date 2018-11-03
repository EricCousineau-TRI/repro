// Purpose: Test if we can forwad-declare aliases.
#include <iostream>
#include <limits>
using namespace std;

// class Alias;

// void blarg(Alias* value);

// class Stuff {
//  public:
//   void things() {
//     cout << "Hello" << endl;
//   }
// };

// using Alias = Stuff;

// void blarg(Alias* value) {
//   value->things();
// }

int main() {
  std::cout << std::numeric_limits<double>::epsilon();
  // Stuff x;
  // blarg(&x);
  return 0;
}
