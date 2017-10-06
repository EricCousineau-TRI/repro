# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module") #, "pybind11_binary")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "py_move",
    srcs = ["py_move.py"],
)

pybind11_module(
    name = "move",
    package_dir = "../..",
)

py_test_simple(
    name = "move_test",
    size = "small",
    deps = [":move"],
)
