cmake_minimum_required(VERSION 3.9)

project(vtk_pybind)

find_package(VTK REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(CTest)
include(pybind11Tools)

pybind11_add_module(vtk_pybind vtk_pybind.cpp)
target_include_directories(vtk_pybind PRIVATE ${VTK_INCLUDE_DIRS})
target_link_libraries(vtk_pybind PRIVATE ${VTK_LIBRARIES})

add_test(NAME vtk_pybind_test COMMAND
    "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/vtk_pybind_test.py")
set_tests_properties(vtk_pybind_test PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:vtk_pybind>:${PYTHONPATH}")
