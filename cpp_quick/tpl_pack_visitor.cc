#include <string>
#include <iostream>

template <typename Visitor>
void pack_visit(Visitor&& visitor) {}

template <typename Visitor, typename T, typename... Ts>
void pack_visit(Visitor&& visitor) {
  visitor.template run<T>();
  pack_visit<Visitor, Ts...>(std::forward<Visitor>(visitor));
};

struct visitor {
  template <typename T>
  void run() {
    std::cout << "\"" << T() << "\"" << std::endl;
  }
};

int main() {
  visitor visit;
  pack_visit<visitor&, int, std::string>(visit);
}
