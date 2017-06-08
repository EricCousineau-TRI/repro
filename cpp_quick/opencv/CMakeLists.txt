cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

add_executable(display_image display_image.cc)
target_link_libraries(display_image ${OpenCV_LIBS})

add_definitions(-std=c++11)
add_executable(manip_repro manip_repro.cc)
target_link_libraries(manip_repro ${OpenCV_LIBS})

add_executable(opencv_animation opencv_animation.cc)
target_link_libraries(opencv_animation ${OpenCV_LIBS})
