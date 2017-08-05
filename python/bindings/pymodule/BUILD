# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module_share")

package(default_visibility = ["//visibility:private"])

pybind11_module_share(
    name = "type_binding",
    package_dir = "..",
    deps = [
        "//python/bindings/pymodule/util:py_relax",
        "@eigen//:eigen",
    ],
)

# Must use absolute path if subpackage (as what `tensorflow` does.
py_library(
    name = "pymodule",  # This is the package's name.
    deps = [
        ":type_binding",
        "//python/bindings/pymodule/sub",
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

py_binary(
    name = "shell",
    srcs = ["shell.py"],
    deps = [":pymodule"],
)
