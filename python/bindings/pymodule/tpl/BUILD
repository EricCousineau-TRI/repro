# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module") #, "pybind11_binary")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "py_tpl",
    srcs = ["py_tpl.py"],
)

pybind11_module(
    name = "scalar_type",
    package_dir = "../..",
    py_deps = [
        ":py_tpl",
    ],
    deps = [
        "//cpp:name_trait",
        "//cpp:simple_converter",
    ],
)

# # How to get this to work???
# pybind11_binary(
#     name = "scalar_type_embed_test",
#     srcs = ["test/scalar_type_embed_test.cc"],
#     pybind11_deps = [
#         ":scalar_type",
#     ],
# )

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

pybind11_module(
    name = "ownership",
    package_dir = "../..",
)

cc_binary(
    name = "ownership_embed",
    srcs = ["_ownership.cc"],
    deps = [
        "//python/pybind11:include",
    ],
)

py_test_simple(
    name = "ownership_test",
    size = "small",
    deps = [":ownership"],
)
