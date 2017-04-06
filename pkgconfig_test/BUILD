load(":cmake_interop.bzl", "cmake_replace_cmd")

includes = ["."]
cflags = ["-I{}".format(include) for include in includes]

reps = {
    "CMAKE_INSTALL_PREFIX": "./bazel-build",
    "_NAME": "lib",
    "_DESCRIPTION": "yup",
    "_URL": "http://awesomesauce/",
    "_VERSION": "1.0",
    "_LIBRARY": "lib",
    "_INCLUDE": ".",
    "_requires": "",
    "_libs": "",
    "_cflags": " ".join(cflags),
    }

genrule(
  name = "pkgconfig",
  srcs = ["template.pc.in"],
  outs = ["lib.pc"],
  cmd = cmake_replace_cmd(reps),
)
