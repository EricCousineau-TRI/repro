// Purpose: See what C++ cares about with C-style casts for overloads and such. 
#include <iostream>

void func(int x) {
  std::cout << "func(int)" << std::endl;
}
typedef void (*f_int)(int);

void func(double x) {
  std::cout << "func(double)" << std::endl;
}
typedef void (*f_double)(double);

int main() {
  f_int a = &func;
  f_double b = &func;

  a(1);
  b(1);

  f_double c = (f_double)a;
  c(1);
  // f_double c2 = static_cast<f_double>(a);
  // c2(1);

  f_double c3 = (f_double)&func;
  c3(1);

  return 0;
}
