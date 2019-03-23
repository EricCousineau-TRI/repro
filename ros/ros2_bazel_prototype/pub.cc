// Based on `examples/rclcpp/minimal_publisher` @ 2dbcf9f
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace {

class Pub : public rclcpp::Node {
 public:
  Pub() : Node("pub"), count_(0)
  {      
    using namespace std::chrono_literals;
    auto publisher = this->create_publisher<std_msgs::msg::String>("topic");
    auto callback = [publisher, mutable int count = 0]() {
      std_msgs::msg::String message;
      message.data = "Hello, world! " + std::to_string(count);
      std::cout << "Pub: " << message.data;
      publisher->publish(message);
    };
    timer_ = this->create_wall_timer(500ms, timer_callback);
  }

 private:
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
