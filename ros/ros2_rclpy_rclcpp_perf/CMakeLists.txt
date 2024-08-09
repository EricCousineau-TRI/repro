cmake_minimum_required(VERSION 3.8)
project(ros2_rclpy_rclcpp_perf)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ExampleStatus.msg"
  "msg/ExampleCommand.msg"
  DEPENDENCIES geometry_msgs
)

ament_package()
# ament_python_install_package(ros2_rclpy_rclcpp_perf)
