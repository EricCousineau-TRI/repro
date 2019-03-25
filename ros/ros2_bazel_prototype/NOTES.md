# Random Exploration Notes

On Ubuntu 18.04

## Bash Stuff

To isolate `bash` from any excess user fluff:

    alias bash-isolate='env -i HOME=$HOME DISPLAY=$DISPLAY SHELL=$SHELL TERM=$TERM USER=$USER PATH=/usr/local/bin:/usr/bin:/bin bash --norc'

Simplifying `ldd` output:

    alias ldd-output-fix="sort | cut -f 1 -d ' ' | sed 's#^\s*##'"

## Bazel Build

Trying to figure out why RMW implementation is so picky for `LD_LIBRARY_PATH`.

See below for doing an overlay with CMake; seems to work fine.

Trying the effectively same thing in Bazel, but without `LD_LIBRARY_PATH`,
yields failures like:

    $ bazel run //:pub_cc
    rmw_fastrtps_cpp
    Failed to find library 'rcl_interfaces__rosidl_typesupport_fastrtps_c'
    terminate called after throwing an instance of 'rclcpp::exceptions::RCLError'
      what():  failed to initialize rcl node: type support not from this implementation, at /tmp/binarydeb/ros-crystal-rmw-fastrtps-cpp-0.6.1/src/rmw_publisher.cpp:81, at /tmp/binarydeb/ros-crystal-rcl-0.6.5/src/rcl/publisher.c:173
    Aborted (core dumped)

Dunno what `not from this implementation` means, as I'm not changing the
implementation, just the library dispatch???

UPDATE: Running prefixed with `strace -s 256`, looks like it checks using the
path of `librmw_impelementation.so`, rather than just checking to see if the
library has already been loaded???

### Trying with direct paths

```
$ strace -s 256 ./bazel-bin/pub_cc 2>&1
execve("./bazel-bin/pub_cc", ["./bazel-bin/pub_cc"], 0x7ffdfb0d4670 /* 75 vars */) = 0
brk(NULL)                               = 0x557e13706000
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/tls/haswell/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat(".../ros2_bazel_prototype/build/hack_overlay/lib/tls/haswell/x86_64", 0x7ffe70a15830) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/tls/haswell/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat(".../ros2_bazel_prototype/build/hack_overlay/lib/tls/haswell", 0x7ffe70a15830) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/tls/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat(".../ros2_bazel_prototype/build/hack_overlay/lib/tls/x86_64", 0x7ffe70a15830) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/tls/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat(".../ros2_bazel_prototype/build/hack_overlay/lib/tls", 0x7ffe70a15830) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/haswell/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat(".../ros2_bazel_prototype/build/hack_overlay/lib/haswell/x86_64", 0x7ffe70a15830) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/haswell/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat(".../ros2_bazel_prototype/build/hack_overlay/lib/haswell", 0x7ffe70a15830) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat(".../ros2_bazel_prototype/build/hack_overlay/lib/x86_64", 0x7ffe70a15830) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, ".../ros2_bazel_prototype/build/hack_overlay/lib/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = 3
```

Is it possible for it to just check for the lib via `ld`, rather than searching
all over the place? ...

### Trying with env paths

```
$ env LD_LIBRARY_PATH=${_ros}/lib strace -s 256 ./bazel-bin/pub_cc
...
execve("./bazel-bin/pub_cc", ["./bazel-bin/pub_cc"], 0x7ffd7e402c30 /* 76 vars */) = 0
brk(NULL)                               = 0x55f9a1aa6000
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f25ba546000
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/tls/haswell/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/opt/ros/crystal/lib/tls/haswell/x86_64", 0x7ffeb1c60560) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/tls/haswell/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/opt/ros/crystal/lib/tls/haswell", 0x7ffeb1c60560) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/tls/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/opt/ros/crystal/lib/tls/x86_64", 0x7ffeb1c60560) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/tls/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/opt/ros/crystal/lib/tls", 0x7ffeb1c60560) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/haswell/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/opt/ros/crystal/lib/haswell/x86_64", 0x7ffeb1c60560) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/haswell/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/opt/ros/crystal/lib/haswell", 0x7ffeb1c60560) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/x86_64/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file or directory)
stat("/opt/ros/crystal/lib/x86_64", 0x7ffeb1c60560) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/opt/ros/crystal/lib/librmw_implementation.so", O_RDONLY|O_CLOEXEC) = 3
```

## C++ CMake Build

Contained hermitic-ish build for
[ros2/examples@2dbcf9f](https://github.com/ros2/examples/tree/2dbcf9f)
(without `setup.bash` - which is really slow???):

**WARNING**: Yeah, this looks real dumb compared to `colcon`. Just trying to see
what's necessary for the Bazel-ness to be happy. You prolly won't ever want
this workflow yourself.

```sh
bash-isolate
cd ros2_bazel_protoype/external  # This repo

# Clear out old stuff.
pwd
rm -rf examples_ws overlay_ws

source ./vars.sh
./clone_and_build.sh

# Er... Some things specify RPATH...
ldd ./examples_ws/build/publisher_lambda | grep ${_ros} | ldd-output-fix
<<EOF
librclcpp.so
librcl.so
librcutils.so
libstd_msgs__rosidl_typesupport_cpp.so
EOF
# ... But others do not?
ldd ./examples_ws/build/publisher_lambda | grep 'not found' | ldd-output-fix
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
env LD_LIBRARY_PATH=${_libs} ldd ./examples_ws/build/publisher_lambda | grep ${_ros} | ldd-output-fix
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
env LD_LIBRARY_PATH=${_overlay}/lib:${_ros}/lib ./publisher_lambda
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
    PYTHONPATH=${_ros_pylib} \
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
    PYTHONPATH=${_ros_pylib} \
    cmake .. \
        -DCMAKE_PREFIX_PATH=${_ros} \
        -DCMAKE_INSTALL_PREFIX=./install
make -j install
# Same as above too :(
ldd install/lib/examples_rclcpp_minimal_publisher/publisher_lambda | grep 'not found' | ldd-output-fix
```
