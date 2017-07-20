// @ref https://stackoverflow.com/questions/7863603/how-to-make-template-rvalue-reference-parameter-only-bind-to-rvalue-reference

#include <iostream>

using namespace std;

template <typename T>
void greedy(T&&) {
  cout << "Greedy" << endl;
}

int main() {
  int x{};
  const int y{};

  greedy(x);
  greedy(y);
  greedy(3);

  return 0;
}
