// Goal: Ensure that we can have a shared singleton with a confined scope.

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>

using std::cout;
using std::endl;
using std::shared_ptr;
using std::weak_ptr;
using std::make_shared;
using std::mutex;

template <typename T, typename Extra = void>
shared_ptr<T> GetScopedSingleton() {
  struct Singleton {
    weak_ptr<T> weak_ref_;
    mutex mutex_;
  };
  static Singleton singleton;
  std::lock_guard<mutex> lock(singleton.mutex_);
  auto instance = singleton.weak_ref_.lock();
  if (!instance) {
    instance = make_shared<T>();
    singleton.weak_ref_ = instance;
  }
  return instance;
}

class Resource {
 public:
    Resource() { cout << "ctor" << endl; }
    ~Resource() { cout << "dtor" << endl; }
};

struct A {};

int main() {
  weak_ptr<Resource> wref;
  weak_ptr<Resource> wref_a;
  cout << "- default: " << wref.use_count() << endl;
  cout << "- A: " << wref_a.use_count() << endl;
  {
    auto ref_1 = GetScopedSingleton<Resource>();
    wref = ref_1;
    cout << "- default: " << wref.use_count() << endl;
    auto ref_2 = GetScopedSingleton<Resource>();
    cout << "- default: " << wref.use_count() << endl;
    {
      auto ref_3 = GetScopedSingleton<Resource>();
      cout << "- default: " << wref.use_count() << endl;
      auto ref_a_1 = GetScopedSingleton<Resource, A>();
      wref_a = ref_a_1;
      cout << "- A: " << wref_a.use_count() << endl;
    }
    cout << "- default: " << wref.use_count() << endl;
    cout << "- A: " << wref_a.use_count() << endl;
  }
  cout << "- default: " << wref.use_count() << endl;

  {
    auto ref_1 = GetScopedSingleton<Resource>();
    wref = ref_1;
    cout << "- default: " << wref.use_count() << endl;
  }
  cout << "- default: " << wref.use_count() << endl;
  return 0;
}
