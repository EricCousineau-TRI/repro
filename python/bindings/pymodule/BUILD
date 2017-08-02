# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load(":pybind11_module.bzl", "pybind11_module")

package(default_visibility = ["//visibility:private"])

pybind11_module(
    name = "typebinding",
    srcs = [
        "_util/py_relax.h",
    ],
    deps = [
        "@eigen//:eigen",
    ],
)

pybind11_module(
    name = "inherit_check",
    srcs = [
        "_util/py_relax.h",
    ],
)

py_library(
    name = "pymodule",
    deps = [
        ":typebinding",
        ":inherit_check",
    ],
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
