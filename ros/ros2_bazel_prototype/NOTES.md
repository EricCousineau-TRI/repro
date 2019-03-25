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

ldd ${_ex_bindir}/publisher_lambda | grep ${_ros} | ldd-output-fix
# - Nothing - good!

# See what ros2 libs we've got - rmw_implementation is in overlay
env LD_LIBRARY_PATH=${_overlay_libdirs} ldd ${_ex_bindir}/publisher_lambda | grep ${_ros} | ldd-output-fix
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
librmw.so
librosgraph_msgs__rosidl_typesupport_cpp.so
librosidl_generator_c.so
librosidl_typesupport_cpp.so
librosidl_typesupport_c.so
libstd_msgs__rosidl_typesupport_cpp.so
libyaml.so
EOF

# Executing:
env LD_LIBRARY_PATH=${_overlay_libdirs} ${_ex_bindir}/publisher_lambda
```

### Try RPATH specification

Tell CMake to bake in library paths, not strip them out? (for easier use w/
Bazel...).

Slightly related (but for Mac SIP stuff):
[ros2#457](https://github.com/ros2/ros2/issues/457)

Try using [kitware wiki RPATH](https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling).

Apply `${_cur}/patches/examples_rpath.patch`...
