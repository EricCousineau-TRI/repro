cmake_minimum_required(VERSION 3.5)
project(cmake_proj)

include(FindPkgConfig)
pkg_check_modules(lib REQUIRED)

# Example of using pkgconfig generated from Bazel
include_directories(${LIB_INCLUDES_DIRS})
add_executable(main main.cc)
target_link_libraries(main ${LIB_LIBRARIES})
