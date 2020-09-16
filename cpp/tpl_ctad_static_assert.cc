#include <type_traits>

template <typename T>
class Merp {
 public:
  static_assert(!std::is_same_v<T, int>, "I don't like ints");

  Merp(T) {}
};

template <typename T>
void func(Merp<T>) {}

int main() {
  Merp(1);
  return 0;
}
