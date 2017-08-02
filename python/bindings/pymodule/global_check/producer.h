#pragma once

#include <sstream>
#include <string>

namespace global_check {

// Goal: Try to recreate issue of duplicate objects when linking with Python
// code.

template <typename T>
std::string Producer(const T& value) {
  static T global{};
  global += value;
  std::ostringstream os;
  os << "Ptr: " << &global << "\nValue: " << global << std::endl;
  return os.str();
}

}  // namespace global_check
