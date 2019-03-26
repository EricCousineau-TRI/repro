load("//tools/skylark:generate_file.bzl", "generate_file")
load("@ros2//:env.bzl", "LD_LIBRARY_PATH", "PYTHONPATH")

_SHIM = r"""
import os
import sys

def prepend_path(key, path):
    os.environ[key] = path + ":" + os.environ.get(key, '')

key = "_ROS2_PY_BAZEL"
if os.environ.get(key, "") != "1":
    os.environ[key] = "1"
    prepend_path("LD_LIBRARY_PATH", {LD_LIBRARY_PATH})
    prepend_path("PYTHONPATH", {PYTHONPATH})

# TODO(eric): Use runfiles resolution.

bin_path = {real}
args = [bin_path] + sys.argv[1:]
os.execv(bin_path, args)
"""

def ros2_py_binary(name, srcs = [], main = None, data = [], deps = [], **kwargs):
    # Introduce shim script to
    real = "_" + name + "_real"
    native.py_binary(
        name = real,
        srcs = srcs,
        main = main,
        data = data,
        deps = deps,
        **kwargs
    )
    prefix = native.package_name()
    if prefix:
        prefix += "/" 
    vars_ = dict(
        LD_LIBRARY_PATH=repr(":".join(LD_LIBRARY_PATH)),
        PYTHONPATH=repr(":".join(PYTHONPATH)),
        real = repr("{}{}".format(prefix, real)),
    )
    shim = "_" + name + "_shim.py"
    generate_file(
        name = shim,
        content = _SHIM.format(**vars_),
    )
    native.py_binary(
        name = name,
        srcs = [shim],
        main = shim,
        data = [":" + real],
        **kwargs
    )
