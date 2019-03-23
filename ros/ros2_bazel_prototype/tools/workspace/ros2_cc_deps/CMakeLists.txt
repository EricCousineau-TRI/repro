cmake_minimum_required(VERSION 3.5)
project(ros2_cc_deps)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(ros2_cc_deps SHARED empty.cc)
ament_target_dependencies(ros2_cc_deps rclcpp std_msgs)

install(TARGETS
  ros2_cc_deps
  DESTINATION lib
)
