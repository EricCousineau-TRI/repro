# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module")

package(default_visibility = ["//visibility:public"])

pybind11_module(
    name = "scalar_type",
    package_dir = "../..",
)

py_library(
    name = "scalar_type",  # This is the package's name.
    deps = [
        ":scalar_type",
        ":func_ptr",
    ],
    visibility = ["//visibility:public"],
)

py_test_simple(
   name = "inherit_check_test",
   size = "small",
   deps = [":inherit_check"],
)

py_test_simple(
   name = "func_ptr_test",
   size = "small",
   deps = [":func_ptr"],
)
