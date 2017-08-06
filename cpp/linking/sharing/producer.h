#include <cstdio>
void produce() {
    static int value = 0;
    ++value;
    printf("  produce: %d (%p)\n", value, (void*)&value);
}

typedef void (*func_t)();
struct funcs_t {
  func_t func;
  func_t func_static;
  funcs_t(func_t a, func_t b)
    : func(a), func_static(b) {}
  void operator()() {
    func();
    func_static();
  }
};
