#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace std;

template <typename... Args>
void unused(Args&&...) {}

template <typename T, typename Del = void(*)(T*)>
unique_ptr<T, Del> release(shared_ptr<T>* p) {
  if (p->use_count() != 1) {
    throw std::runtime_error("Must have unique shared_ptr");
  }
  Del* pdeleter = std::get_deleter<Del>(*p);
  assert(pdeleter);
  Del orig_deleter = *pdeleter;
  Del null_delete = [](T* raw) {
    unused(raw);
  };
  *pdeleter = null_delete;

  unique_ptr<T, Del> out(p->get(), orig_deleter);
  // shared_ptr<T> shallow_copy(nullptr, null_delete);
  // shallow_copy.swap(*p);
  // Get and destroy shared_ptr, but keep instance alive.
  return out;
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
  unique_ptr<A> raw = nullptr;
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

  return 0;
}
