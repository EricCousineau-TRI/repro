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

### Odd RMW loading behavior

Without `LD_LIBRARY_PATH`:

```
$ strace -s 256 ./bazel-bin/pub_cc 2>&1
# See ./logs/strace_no_env.txt
```

Is it possible for it to just check for the lib via `ld`, rather than searching
all over the place? ...

With it:

```
$ env LD_LIBRARY_PATH=${_overlay_libdirs} strace -s 256 ./bazel-bin/pub_cc
# See ./logs/strace_env.txt
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
