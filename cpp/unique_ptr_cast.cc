#include <iostream>
#include <memory>

using namespace std;

struct Base {
  virtual ~Base() {}
  virtual void stuff() = 0;
};

struct Derived : public Base {
  virtual void stuff() {
    cout << "Derived::stuff" << endl;
  }
};

int main() {
  unique_ptr<Derived> derived(new Derived());
  unique_ptr<Base> base = std::move(derived);
  base->stuff();
  return 0;
}
