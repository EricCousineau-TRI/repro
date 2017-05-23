cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

add_executable(display_image display_image.cc)
target_link_libraries(display_image ${OpenCV_LIBS})
