cmake_minimum_required(VERSION 3.9)

project(vtk_pybind)

find_package(VTK REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(CTest)
include(pybind11Tools)

pybind11_add_module(vtk_pybind_example vtk_pybind_example.cpp)
target_include_directories(vtk_pybind_example PRIVATE ${VTK_INCLUDE_DIRS})
target_link_libraries(vtk_pybind_example PUBLIC ${VTK_LIBRARIES})

# N.B. This is NOT generic! Dunno how to simplify...
set(vtk_python_dir ${VTK_DIR}/../../python3.6/site-packages)
set(vtk_lib_dir ${VTK_DIR}/../..)

add_test(NAME vtk_pybind_example_test COMMAND
    "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/vtk_pybind_example_test.py")
# Blech...
set_tests_properties(vtk_pybind_example_test PROPERTIES
    ENVIRONMENT
        "LD_LIBRARY_PATH=${vtk_lib_dir}:$ENV{LD_LIBRARY_PATH};PYTHONPATH=$<TARGET_FILE_DIR:vtk_pybind_example>:${vtk_python_dir}:$ENV{PYTHONPATH}")
