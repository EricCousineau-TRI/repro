# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "python_repository")

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "_typebinding.so",
    srcs = [
        "_typebinding.cc",
        "_util/py_relax.h",
    ],
    copts = [
        "-Wno-#warnings",
        "-Wno-cpp",
        "-Wno-unknown-warning-option",
    ],
    # Could put `typebinding.py` as `data` field, but then it may avoid `pylint()`
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "@eigen//:eigen",
        "@pybind11//:pybind11",
    ],
)

py_library(
    name = "pymodule",
    srcs = [
        "__init__.py",
        "typebinding.py",
    ],
    data = [
        ":_typebinding.so",
    ],
    imports = [".."],
    visibility = ["//visibility:public"],
)

py_test(
   name = "typebinding_test",
   size = "small",
   srcs = ["test/testTypeBinding.py"],
   main = "test/testTypeBinding.py",
   tags = [],
   deps = [":pymodule"],
)
