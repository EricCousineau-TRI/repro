#include <iostream>
#include <functional>
#include <string>

using namespace std;

std::function<void()> GetFunc(const string& x) {
  return [&x]() {
    cout << "x: " << x << endl;
  };
}

int main() {
  string x = "Works";
  auto func = GetFunc(x);
  func();

  GetFunc("Sketch, but might work")();

  std::function<void()> func3;
  {
    func3 = GetFunc("Will corrupt");
  }
  func3();
  return 0;
}

