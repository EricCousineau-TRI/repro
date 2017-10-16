#include <iostream>
#include <memory>
#include <string>

using namespace std;

void do_stuff(shared_ptr<string> y) {
  cout << y.get() << endl;
  cout << *y << endl;
}

int main() {
  do_stuff(make_unique<string>("Hello world"));

  return 0;
}
