#include <rclcpp/rclcpp.hpp>
#include "ros/ros.h"

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  ros::init(argc, argv, "blipper");
  return 0;
}
