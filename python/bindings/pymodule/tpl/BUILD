# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "tpl_def",
    srcs = ["tpl_def.py"],
)

pybind11_module(
    name = "scalar_type",
    package_dir = "../..",
    deps = [
        ":tpl_def",
        "//cpp:name_trait",
    ]
)

py_library(
    name = "tpl",  # This is the package's name.
    deps = [
        ":scalar_type",
    ],
    visibility = ["//visibility:public"],
)

py_test_simple(
    name = "scalar_type_test",
    size = "small",
    deps = [":scalar_type"],
)
