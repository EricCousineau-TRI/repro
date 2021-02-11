Taken from: <https://github.com/RobotLocomotion/drake/pull/12061/files>

# Hacks for Testing latest `libsdformat` stuff

## Deps

For Ubuntu Bionic (may need ROS repos...)

    sudo apt install \
        libtinyxml2-dev python3-vcstool python3-colcon-common-extensions

Then update repositories (and record the exact versions afterwards):

    (
        set -eux
        vcs import ./ < ./meta/vcs.repo
        vcs export --exact ./ > ./meta/vcs-exact.repo
    )

First, test without Bazel overhead:

    colcon build
    colcon test
    # Or just go the build package, and run `make test` / `ctest`.
