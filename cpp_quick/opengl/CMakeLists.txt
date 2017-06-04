cmake_minimum_required(VERSION 3.5)
project(opengl_test)

# @ref https://stackoverflow.com/questions/9460242/how-to-compile-glut-opengl-project-with-cmake-and-kdevelop-in-linux
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

set(SRCS
    ${CMAKE_BINARY_DIR}/blender.c)
# https://stackoverflow.com/questions/32512872/cmake-executable-with-auto-generated-sources
add_custom_command(
    OUTPUT ${SRCS}
    COMMAND ./sync.sh ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(opengl_test ${SRCS})

target_link_libraries(opengl_test ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} m)
