# `ros2_bazel_prototype`

Purpose: See what options there are to consume ROS2 C++ and Python packages
from Bazel.

## Status

**Work In Progress** - ain't nothing useful been done yet

## Motivation

[drake](https://drake.mit.edu) and some teams at TRI use Bazel.

For ROS1, we used `pkg-config` to get the necessary C++ build bits; for Python,
we just used the existing FHS install tree.

For ROS2, though, `pkg-config` files are no longer emitted.

## Existing

There exists [colcon-bazel](https://github.com/colcon/colcon-bazel) as a
plugin, but it seems:
* To be geared towards dispatching commands (`build`, `run`, `test`),
not for emitting artifacts for Bazel to consume other `colcon`-built
packages.
* Not to provide Bazel / Skylark rules to emit artifacts for other build
systems to use.

## Proposal

For now, probably just run `cmake` w/ some `colcon` / `ament` instrumentation,
write some config files, and then have Bazel scoop those up in a
`repository_rule`.

Could maybe use [bazelbuild/rules_foreign_cc](https://github.com/bazelbuild/rules_foreign_cc)
to dispatch to CMake; however, I'm not sure how it permits consuming the
artifacts.

UPDATE: `rules_foreign_cc` is uber nice, but I dunno how to consume modern
CMake interface targets. For this reason, I kludged `cmake_cc.bzl`.

See [NOTES.md](./NOTES.md) for random exploration stuff.
