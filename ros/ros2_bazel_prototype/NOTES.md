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
