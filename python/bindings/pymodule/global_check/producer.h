#pragma once

#include <sstream>
#include <string>

namespace global_check {

// Goal: Try to recreate issue of duplicate objects when linking with Python
// code.

struct Impl {
  static double global;
};
double Impl::global{};

template <typename T>
std::string Producer(const T& value) {
  T& global = Impl::global;
  global += value;
  std::ostringstream os;
  os << "Ptr: " << &global << "\nValue: " << global << std::endl;
  return os.str();
}

}  // namespace global_check
