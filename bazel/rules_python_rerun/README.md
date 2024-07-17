# `rerun` using `rules_python`

Trying <https://rerun.io/>

Modeled after
https://github.com/RobotLocomotion/drake-blender

## Issue

```sh
$ cd rules_python_rerun
$ bazel build //:example
$ bazel-bin/example
${PWD}
{runfiles}
{runfiles}/pip_deps_attrs/site-packages
{runfiles}/pip_deps_numpy/site-packages
{runfiles}/pip_deps_pillow/site-packages
{runfiles}/pip_deps_pyarrow/site-packages
{runfiles}/pip_deps_typing_extensions/site-packages
{runfiles}/pip_deps_rerun_sdk/site-packages
{runfiles}/bazel_tools
{runfiles}/meshcat_stuff
{runfiles}/pip_deps_attrs
{runfiles}/pip_deps_numpy
{runfiles}/pip_deps_pillow
{runfiles}/pip_deps_pyarrow
{runfiles}/pip_deps_rerun_sdk
{runfiles}/pip_deps_typing_extensions
/usr/lib/python310.zip
/usr/lib/python3.10
/usr/lib/python3.10/lib-dynload
/usr/local/lib/python3.10/dist-packages
/usr/lib/python3/dist-packages
/usr/lib/python3.10/dist-packages
Traceback (most recent call last):
  File "/home/eacousineau/proj/tri/repo/repro/bazel/rules_python_rerun/bazel-bin/example.runfiles/meshcat_stuff/example.py", line 28, in <module>
    main()
  File "/home/eacousineau/proj/tri/repo/repro/bazel/rules_python_rerun/bazel-bin/example.runfiles/meshcat_stuff/example.py", line 21, in main
    import rerun as rr
ModuleNotFoundError: No module named 'rerun'

# inspecting
$ cd $(bazel info output_base)/
# this is not on python path
$ ls external/pip_deps_rerun_sdk/site-packages/rerun_sdk
__init__.py  rerun  rerun_cli
```
