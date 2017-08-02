# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load(":pybind11_module.bzl", "pybind11_module", "drake_py_test")

package(default_visibility = ["//visibility:private"])

pybind11_module(
    name = "type_binding",
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
        ":type_binding",
        ":inherit_check",
    ],
    visibility = ["//visibility:public"],
)

drake_py_test(
   name = "type_binding_test",
   size = "small",
   deps = [":type_binding"],
)

drake_py_test(
   name = "inherit_check_test",
   size = "small",
   deps = [":inherit_check"],
)
