cmake_minimum_required(VERSION 2.8.12)
project(pybind_sphinx_init CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(example SHARED example_py.cc)

add_custom_target(
    build_sphinx ${CMAKE_COMMAND} -E
        env PYTHONPATH=$<TARGET_FILE_DIR:example>:${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/build_sphinx.sh
            ${PYTHON_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc
)
