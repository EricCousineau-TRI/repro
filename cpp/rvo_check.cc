#include <iostream>
#include <type_traits>

#include "name_trait.h"
#include "type_pack.h"

template <typename T>
struct Count {
  static int num_constructed;
  int value{100};

  Count() { num_constructed++; }
  Count(const Count&) { num_constructed++; }
  Count(Count&&) { num_constructed++; }
};
template <typename T>
int Count<T>::num_constructed = 0;

struct A : public Count<A> {};

struct B : public Count<B> {};

struct C : public Count<C> {
  using Base = Count<C>;
  // Allow implicit conversion.
  using Base::Base;
  C(const B&) : Base() {}
};

template <typename T>
auto GetTypeTag() {
  if constexpr (std::is_same_v<T, B>) {
    return type_tag<C>{};
  } else {
    return type_tag<const T&>{};
  }
}

template <typename T>
using get_type = typename decltype(GetTypeTag<T>())::type;

template <typename T>
get_type<T> MaybeConvert(const T& value) { return value; }

template <typename T>
void Check(const T& x) {
  const auto& y = MaybeConvert(x);
  if (y.value != 100) {
    throw std::runtime_error("Bad: " + std::to_string(y.value));
  }
}

void print_constructed_and_reset() {
  std::cout
      << "num_constructed: "
      << "A: " << A::num_constructed << ", "
      << "B: " << B::num_constructed << ", "
      << "C: " << C::num_constructed << "\n\n\n";
  A::num_constructed = 0;
  B::num_constructed = 0;
  C::num_constructed = 0;
}

int main() {
  EVAL(Check(A{}));
  print_constructed_and_reset();
  EVAL(Check(B{}));
  print_constructed_and_reset();
  EVAL(Check(C{}));
  print_constructed_and_reset();
  return 0;
}

/**
>>> Check(A{});

num_constructed: A: 1, B: 0, C: 0


>>> Check(B{});

num_constructed: A: 0, B: 1, C: 1


>>> Check(C{});

num_constructed: A: 0, B: 0, C: 1

 */
