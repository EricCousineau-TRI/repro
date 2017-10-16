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
std::pair<std::string, T> Producer(const T& value) {
  T& global = Impl::global;
  global += value;
  std::ostringstream os;
  os << "Ptr: " << &global;
  return std::make_pair(os.str(), global);
}

inline std::pair<std::string, double> ProducerB(double value) {
  static double global{};
  global += value;
  std::ostringstream os;
  os << "Ptr: " << &global;
  return std::make_pair(os.str(), global);
}

}  // namespace global_check
