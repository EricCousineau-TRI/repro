# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "_dup.so",
    srcs = [
        "_dup.cc",
    ],
    copts = [
        "-Wno-#warnings",
        "-Wno-cpp",
        "-Wno-unknown-warning-option",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "@pybind11//:pybind11",
    ],
)

py_library(
    name = "sub",
    srcs = [
        "__init__.py",
    ],
    data = [
        ":_dup.so",
    ],
    imports = [".."],
    visibility = ["//visibility:public"],
)
