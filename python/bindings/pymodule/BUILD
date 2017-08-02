# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module")

package(default_visibility = ["//visibility:private"])

pybind11_module(
    name = "type_binding",
    package_dir = "..",
    srcs = [
        "_util/py_relax.h",
    ],
    deps = [
        "@eigen//:eigen",
    ],
)

py_library(
    name = "pymodule",  # This is the package's name.
    deps = [
        ":type_binding",
        "//python/bindings/pymodule/sub",  # Must use absolute path if subpackage.
    ],
    visibility = ["//visibility:public"],
)

py_test_simple(
   name = "type_binding_test",
   size = "small",
   deps = [":type_binding"],
)

py_test_simple(
    name = "pymodule_test",
    size = "small",
    deps = [":pymodule"],
)
