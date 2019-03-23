cmake_minimum_required(VERSION 3.5)
project(ros2_cc_deps)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_cmake_export_interfaces REQUIRED)

# TODO: Can Bazel `cmake_external` be taught to parse CMake interface targets
# and consume their include info?
# Maybe? https://answers.ros.org/question/305991/how-to-cleanly-handle-dependencies-in-ament/
# https://github.com/dirk-thomas/ament_examples/blob/75d6209e4e2db33ad957a2abd1a3a7e102068863/example/CMakeLists.txt

add_library(ros2_cc_deps SHARED empty.cc)
ament_target_dependencies(ros2_cc_deps rclcpp std_msgs)
# What is this supposed to produce???
ament_export_interfaces(ros2_cc_deps)

install(
  TARGETS ros2_cc_deps
  EXPORT ros2_cc_deps
  DESTINATION lib
)
