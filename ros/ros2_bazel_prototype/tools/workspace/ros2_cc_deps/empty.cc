#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace empty {

namespace {

std::size_t accumulate(std::size_t h1, std::size_t h2) {
  return h1 ^ (h2 << 1);
}

void accumulate_to(std::size_t& h1, std::size_t h2) {
  h1 = accumulate(h1, h2);
}

} // namespace

std::size_t prevent_optimization_from_going_thanos_on_this_obj_code() {
  std::size_t result{};
  // Use some random symbols.
  accumulate_to(result, typeid(rclcpp::Node).hash_code());
  accumulate_to(result, typeid(std_msgs::msg::String).hash_code());
  return result;
}

}  // namespace
