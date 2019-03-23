# Random Exploration Notes

## Setup

To isolate `bash` from any excess user fluff:

    alias bash-isolate='env -i HOME=$HOME DISPLAY=$DISPLAY SHELL=$SHELL TERM=$TERM USER=$USER PATH=/usr/local/bin:/usr/bin:/bin bash --norc'

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
env LD_LIBRARY_PATH=${_ros}/lib ./publisher_lambda
```

TODO:

*   Tell CMake to bake in library paths, not strip them out? (for easier use
w/ Bazel)
    * Meh for now. Slightly related (but for Mac SIP stuff):
    [ros2#457](https://github.com/ros2/ros2/issues/457)
    * Tried using [kitware wiki RPATH](https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling), but as flags:

```sh
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
ldd install/lib/examples_rclcpp_minimal_publisher/publisher_lambda | grep 'not found'
<<EOF
    librcl_interfaces__rosidl_typesupport_cpp.so => not found
    librmw_implementation.so => not found
    librmw.so => not found
    librcl_yaml_param_parser.so => not found
    librosgraph_msgs__rosidl_typesupport_cpp.so => not found
    librcl_interfaces__rosidl_typesupport_c.so => not found
    librcl_interfaces__rosidl_generator_c.so => not found
    librmw_implementation.so => not found
    librmw.so => not found
    librosidl_generator_c.so => not found
    librcl_logging_noop.so => not found
    librosidl_typesupport_cpp.so => not found
EOF
# But finds other libraries... Why???
```
