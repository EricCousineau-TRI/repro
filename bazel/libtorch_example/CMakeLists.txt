cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(libtorch_test libtorch_test.cc)
target_link_libraries(libtorch_test "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_test PROPERTY CXX_STANDARD 17)
