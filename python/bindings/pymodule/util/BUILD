# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module")

package(default_visibility = ["//python/bindings/pymodule:__subpackages__"])

cc_library(
    name = "py_relax",
    srcs = [
        "py_relax.h",
    ],
    deps = [
        "@eigen//:eigen",
    ],
)

py_library(
    name = "share_symbols",
    srcs = [
        "share_symbols.py",
    ],
)
