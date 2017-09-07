#include <iostream>
#include <memory>
#include <stdexcept>

using namespace std;

template <typename... Args>
void unused(Args&&...) {}

template <typename T>
T* release(shared_ptr<T>* p) {
  if (p->use_count() != 1) {
    throw std::runtime_error("Must have sole shared_ptr");
  }
  auto null_delete = [](T* raw) {
    unused(raw);
  };
  shared_ptr<T> shallow_copy(nullptr, null_delete);
  shallow_copy.swap(*p);
  // Get and destroy shared_ptr, but keep instance alive.
  return shallow_copy.get();
}


struct A {
  A() {
    cout << "A()" << endl;
  }
  ~A() {
    cout << "~A()" << endl;
  }
};

int main() {
  A* raw = nullptr;
  {
    cout << "stack" << endl;
    A x;
  }
  {
    shared_ptr<A> a(new A());
    raw = release(&a);
    cout << "released" << endl;
  }
  cout << "finish" << endl;
  delete raw;

  return 0;
}
