# Random Exploration Notes

On Ubuntu 18.04

## Bash Stuff

To isolate `bash` from any excess user fluff:

    alias bash-isolate='env -i HOME=$HOME DISPLAY=$DISPLAY SHELL=$SHELL TERM=$TERM USER=$USER PATH=/usr/local/bin:/usr/bin:/bin bash --norc'

Simplifying `ldd` output:

    alias ldd-output-fix="sort | cut -f 1 -d ' ' | sed 's#^\s*##'"

## C++ CMake Build

Contained hermitic-ish build for
[ros2/examples@2dbcf9f](https://github.com/ros2/examples/tree/2dbcf9f)
(without `setup.bash` - which is really slow???):

```sh
bash-isolate
cd $(mktemp -d)
pwd
git clone https://github.com/ros2/examples
cd examples
git checkout 2dbcf9f
cd rclcpp/minimal_publisher
rm -rf build && mkdir build && cd build
_ros=/opt/ros/crystal
_py=3.6
env PYTHONPATH=${_ros}/lib/python${_py}/site-packages \
    cmake .. -DCMAKE_PREFIX_PATH=${_ros}
make
# Er... Some things specify RPATH...
ldd ./publisher_lambda | grep ${_ros} | ldd-output-fix
<<EOF
librclcpp.so
librcl.so
librcutils.so
libstd_msgs__rosidl_typesupport_cpp.so
EOF
# ... But others do not?
ldd ./publisher_lambda | grep 'not found' | ldd-output-fix
<<EOF
librcl_interfaces__rosidl_generator_c.so
librcl_interfaces__rosidl_typesupport_cpp.so
librcl_interfaces__rosidl_typesupport_c.so
librcl_logging_noop.so
librcl_yaml_param_parser.so
librmw_implementation.so
librmw_implementation.so
librmw.so
librmw.so
librosgraph_msgs__rosidl_typesupport_cpp.so
librosidl_generator_c.so
librosidl_typesupport_cpp.so
EOF
# The following of course works.
env LD_LIBRARY_PATH=${_ros}/lib ldd ./publisher_lambda | grep ${_ros} | ldd-output-fix
<<EOF
libbuiltin_interfaces__rosidl_generator_c.so
librclcpp.so
librcl_interfaces__rosidl_generator_c.so
librcl_interfaces__rosidl_typesupport_cpp.so
librcl_interfaces__rosidl_typesupport_c.so
librcl_logging_noop.so
librcl.so
librcl_yaml_param_parser.so
librcutils.so
librmw_implementation.so
librmw.so
librosgraph_msgs__rosidl_typesupport_cpp.so
librosidl_generator_c.so
librosidl_typesupport_cpp.so
librosidl_typesupport_c.so
libstd_msgs__rosidl_typesupport_cpp.so
libyaml.so
EOF
# Executing:
env LD_LIBRARY_PATH=${_ros}/lib ./publisher_lambda
```

### Try RPATH specification

Tell CMake to bake in library paths, not strip them out? (for easier use w/
Bazel...).

Slightly related (but for Mac SIP stuff):
[ros2#457](https://github.com/ros2/ros2/issues/457)

Try using [kitware wiki RPATH](https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling), but as flags:

```sh
cd build && rm -rf ./*
env \
    LD_LIBRARY_PATH=${_ros}/lib \
    PYTHONPATH=${_ros}/lib/python${_py}/site-packages \
    cmake .. \
        -DCMAKE_PREFIX_PATH=${_ros} \
        -DCMAKE_INSTALL_PREFIX=./install \
        -DCMAKE_SKIP_BUILD_RPATH=FALSE \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=False \
        -DCMAKE_INSTALL_RPATH="./install/lib;${_ros}/lib" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE
make -j install
# The following two are exactly the same as above???
ldd install/lib/examples_rclcpp_minimal_publisher/publisher_lambda | grep ${_ros} | ldd-output-fix
ldd install/lib/examples_rclcpp_minimal_publisher/publisher_lambda | grep 'not found' | ldd-output-fix
```

* Try full gamut:

```sh
cd build && rm -rf ./*
( cd ../../.. && git apply - <<EOF
diff --git a/rclcpp/minimal_publisher/CMakeLists.txt b/rclcpp/minimal_publisher/CMakeLists.txt
index 3344f14..ca8830f 100644
--- a/rclcpp/minimal_publisher/CMakeLists.txt
+++ b/rclcpp/minimal_publisher/CMakeLists.txt
@@ -1,6 +1,16 @@
 cmake_minimum_required(VERSION 3.5)
 project(examples_rclcpp_minimal_publisher)
 
+# From: https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling#always-full-rpath
+SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
+SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
+SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
+SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
+LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
+IF("${isSystemDir}" STREQUAL "-1")
+   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
+ENDIF("${isSystemDir}" STREQUAL "-1")
+
 # Default to C++14
 if(NOT CMAKE_CXX_STANDARD)
   set(CMAKE_CXX_STANDARD 14)
EOF
)

env \
    PYTHONPATH=${_ros}/lib/python${_py}/site-packages \
    cmake .. \
        -DCMAKE_PREFIX_PATH=${_ros} \
        -DCMAKE_INSTALL_PREFIX=./install
make -j install
# Same as above too :(
ldd install/lib/examples_rclcpp_minimal_publisher/publisher_lambda | grep 'not found' | ldd-output-fix
```
