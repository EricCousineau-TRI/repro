#include <string>
#include <iostream>

using namespace std;

// https://stackoverflow.com/a/38617219/7829525
#define TYPE_SUPPORTS(ClassName, Expr)                         \
  template<typename U>                                         \
  struct ClassName                                             \
  {                                                            \
   private:                                                    \
    template<typename>                                         \
    static constexpr std::false_type test(...);                \
                                                               \
    template<typename T = U>                                   \
    static decltype((Expr), std::true_type{}) test(int) ;      \
                                                               \
   public:                                                     \
    static constexpr bool value = decltype(test<U>(0))::value; \
  };

void g() {}

void f() {
  // Can use this for SFINAE
  decltype((g(), 2)) x{};
  cout << x << endl;
}

TYPE_SUPPORTS(has_run, std::declval<T>().template run<int>());

template <typename Visitor>
void pack_visit(Visitor&& visitor) {}

template <typename Visitor, typename T, typename... Ts>
void pack_visit(Visitor&& visitor) {
  visitor.template run<T>();
  pack_visit<Visitor, Ts...>(std::forward<Visitor>(visitor));
};

// To infer caller type
template <typename... Ts>
struct pack_visitor {
  template <typename Visitor>
  static void run(Visitor&& visitor) {
    pack_visit<Visitor, Ts...>(std::forward<Visitor>(visitor));
  }
};

struct visitor {
  template <typename T>
  void run() {
    std::cout << "\"" << T() << "\"" << std::endl;
  }
};

int main() {
  f();
  visitor visit;
  pack_visit<visitor&, int, std::string>(visit);
  pack_visitor<int, std::string>::run(visit);
}
