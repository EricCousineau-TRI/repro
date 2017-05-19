// Goal: Ensure that we can hide internal implentation details.

#include <functional>
#include <iostream>
#include <memory>

using std::cout;
using std::endl;
using std::shared_ptr;
using std::weak_ptr;
using std::make_shared;

template <typename T, typename Extra = void>
class ScopedSingleton {
 public:
  typedef shared_ptr<T> ResourcePtr;

  static shared_ptr<T> instance() {
    static weak_ptr<T> ref;
    auto instance = ref.lock();
    if (!instance) {
      instance = make_shared<T>();
      ref = instance;
    }
    return instance;
  }

 private:
  typedef weak_ptr<T> ResourceWeakPtr;
};


class Resource {
 public:
    Resource() { cout << "ctor" << endl; }
    ~Resource() { cout << "dtor" << endl; }
};

struct A {};

int main() {
  weak_ptr<Resource> wref;
  {
    auto ref_1 = ScopedSingleton<Resource>::instance();
    wref = ref_1;
    cout << wref.use_count() << endl;
    auto ref_2 = ScopedSingleton<Resource>::instance();
    cout << wref.use_count() << endl;
    {
      auto ref_a_1 = ScopedSingleton<Resource, A>::instance();
    cout << wref.use_count() << endl;
      auto ref_3 = ScopedSingleton<Resource>::instance();
    cout << wref.use_count() << endl;
    }
  }
  cout << wref.use_count() << endl;
  return 0;
}
