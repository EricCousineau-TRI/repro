#include "producer_linkshared.h"

namespace global_check {

std::pair<std::string, double> ProducerB(double value) {
  static double global{};
  global += value;
  std::ostringstream os;
  os << "Ptr: " << &global;
  return std::make_pair(os.str(), global);
}

}  // namespace global_check
