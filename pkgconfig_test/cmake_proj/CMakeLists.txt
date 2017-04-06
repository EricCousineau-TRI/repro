project(cmake_proj)

find_package(lib)

# Example of using pkgconfig generated from Bazel
add_binary(main main.cc)
target_link_libraries(main lib)
