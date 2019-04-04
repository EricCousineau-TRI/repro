cmake_minimum_required(VERSION 3.5)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(basic_shapes basic_shapes.cc)
ament_target_dependencies(basic_shapes rclcpp visualization_msgs)
