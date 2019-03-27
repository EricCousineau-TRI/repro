cmake_minimum_required(VERSION 3.5)

find_package(ament_cmake REQUIRED)
find_package(catkin REQUIRED roscpp)
find_package(rclcpp REQUIRED)

add_executable(i_am_special i_am_special.cc)
ament_target_dependencies(i_am_special rclcpp)
target_link_libraries(i_am_special ${catkin_LIBRARIES})
target_include_directories(i_am_special PRIVATE ${catkin_INCLUDE_DIRS})
