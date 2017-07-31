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

cc_binary(
    name = "_inherit_check.so",
    srcs = [
        "_inherit_check.cc",
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
        "@pybind11//:pybind11",
    ],
)

py_library(
    name = "pymodule",
    srcs = [
        "__init__.py",
        "typebinding.py",
        "inherit_check.py",
    ],
    data = [
        ":_typebinding.so",
        ":_inherit_check.so",
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

py_test(
   name = "inherit_check_test",
   size = "small",
   srcs = ["test/testInheritCheck.py"],
   main = "test/testInheritCheck.py",
   tags = [],
   deps = [":pymodule"],
)
