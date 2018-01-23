#include <functional>
#include <iostream>
#include <type_traits>

#include "cpp/name_trait.h"
#include "cpp/wrap_function.h"

using namespace std;

// Base case: Pass though.
template <typename T, typename = void>
struct ensure_ptr : public wrap_arg_default<T> {};

// TODO(eric.cousineau): When using in pybind, ensure this doesn't actually
// apply to types with non-trivial type casters.
// Looks like `std::is_base<type_caster_base<T>, type_caster<T>>` is the ticket?

// NOTE: Should still ensure that `std::function<>` takes precedence.

template <typename T>
struct ensure_ptr<const T*, std::enable_if_t<!std::is_same<T, int>::value>> {
  static const T* wrap(const T* arg) {
    cout << "<const T*> wrap: " << nice_type_name<const T*>() << endl;
    return arg;
  }
  static const T* unwrap(const T* arg) {
    cout << "<const T*> unwrap: " << nice_type_name<const T*>() << endl;
    return arg;
  }
};

template <typename T>
struct ensure_ptr<const T&> {
  static const T* wrap(const T& arg) {
    cout << "<const T&> wrap: " << nice_type_name<const T&>() << endl;
    return &arg;
  }
  static const T& unwrap(const T* arg) {
    cout << "<const T&> unwrap: " << nice_type_name<const T&>() << endl;
    return *arg;
  }
};

// Reference case: Convert to pointer.
template <typename T>
struct ensure_ptr<T&> {
  static T* wrap(T& arg) {
    cout << "<T&> wrap: " << nice_type_name<T&>() << endl;
    return &arg;
  }
  static T& unwrap(T* arg) {
    cout << "<T&> unwrap: " << nice_type_name<T&>() << endl;
    return *arg;
  }
};

template <typename Func>
auto EnsurePtr(Func&& func) {
  return WrapFunction<ensure_ptr>(std::forward<Func>(func));
}


struct MoveOnlyValue {
  MoveOnlyValue() = default;
  MoveOnlyValue(const MoveOnlyValue&) = delete;
  MoveOnlyValue(MoveOnlyValue&&) = default;
  int value{};
};

void Func_1(int value) {}
int* Func_2(int& value) { value += 1; return &value; }
const int& Func_3(const int& value) { return value; }
void Func_4(MoveOnlyValue value) {}
void Func_5(const int* value) {}

void Func_6(int& value, std::function<void (int&)> callback) {
  callback(value);
}

int& Func_7(int& value, const std::function<int& (int&)>& callback) {
  return callback(value);
}

class MyClass {
 public:
  static void Func(MoveOnlyValue&& value) {}
  void Method(MoveOnlyValue& value) { value.value += 2; }
  void Method_2(MoveOnlyValue& value) const { value.value += 3; }
};

struct MoveOnlyFunctor {
  MoveOnlyFunctor() {}
  MoveOnlyFunctor(const MoveOnlyFunctor&) = delete;
  MoveOnlyFunctor(MoveOnlyFunctor&&) = default;
  // Cannot overload operator(), as it's ambiguous.
  // NOTE: Could make this mutable, but doesn't make sense to?
  void operator()(MoveOnlyValue& value) const {
    value.value += 4;
  }
};

struct ConstFunctor {
  void operator()(MoveOnlyValue& value) const { value.value += 5; }
};

#define CHECK(expr) EVAL(expr); cout << "v.value = " << v.value << endl << endl

int main() {
  MoveOnlyValue v{10};
  CHECK(EnsurePtr(Func_1)(v.value));
  CHECK(cout << *EnsurePtr(Func_2)(&v.value));
  CHECK(cout << *EnsurePtr(Func_3)(&v.value));
  CHECK(EnsurePtr(Func_4)(MoveOnlyValue{}));
  CHECK(EnsurePtr(Func_5)(&v.value));
  auto void_ref = [](int& value) {
    value += 10;
  };
  CHECK(EnsurePtr(void_ref)(&v.value));

  CHECK(EnsurePtr(MyClass::Func)(MoveOnlyValue{}));
  MyClass c;
  const MyClass& c_const{c};
  CHECK(EnsurePtr(&MyClass::Method)(&c, &v));
  CHECK(EnsurePtr(&MyClass::Method_2)(&c_const, &v));

  MoveOnlyFunctor f;
  CHECK(EnsurePtr(std::move(f))(&v));
  ConstFunctor g;
  CHECK(EnsurePtr(g)(&v));
  const ConstFunctor& g_const{g};
  CHECK(EnsurePtr(g_const)(&v));

  // Callback.
  CHECK(EnsurePtr(Func_6)(&v.value, EnsurePtr(void_ref)));

  // Callback with return.
  auto get_ref = [](int& value) -> int& {
    value += 100;
    return value;
  };
  CHECK(cout << *EnsurePtr(Func_7)(&v.value, EnsurePtr(get_ref)));

  // Nested callback.
  auto get_ref_nested = [get_ref](int& value,
      std::function<int& (int&, const std::function<int& (int&)>&)> func) -> auto& {
    value += 1000;
    return func(value, get_ref);
  };
  CHECK(cout << *EnsurePtr(get_ref_nested)(&v.value, EnsurePtr(Func_7)));

  return 0;
}
