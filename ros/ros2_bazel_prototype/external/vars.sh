#!/bin/bash

# @file
# To be sourced for use in other scripts / sessions.
# Should NOT dump any environment variables anywhere.

alias bash-isolate='env -i HOME=$HOME DISPLAY=$DISPLAY SHELL=$SHELL TERM=$TERM USER=$USER PATH=/usr/local/bin:/usr/bin:/bin bash --norc'
mkcd() { mkdir "$@" && cd "${!#}"; }
alias ldd-output-fix="sort | cut -f 1 -d ' ' | sed 's#^\s*##'"

_ros=/opt/ros/crystal
_ros_pylib=${_ros}/lib/python3.6/site-packages
_cur=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_overlay=${_cur}/overlay_ws/install

_overlay_libdirs=${_overlay}/lib:${_ros}/lib
_ex_bindir=${_cur}/examples_ws/install/lib/examples_rclcpp_minimal_publisher
