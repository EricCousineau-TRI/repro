# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module", "pybind_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cpp_template",
    srcs = ["cpp_template.py"],
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

cc_library(
    name = "cpp_template_types_cc",
    srcs = [
        "cpp_types.h",
        "cpp_types.cc",
    ],
    deps = [
        "//cpp:type_pack",
        "//cpp:name_trait",
        "//python/pybind11:include",
    ],
)

pybind11_module(
    name = "cpp_types",
    package_dir = "../..",
    deps = [
        ":cpp_template_types_cc",
    ],
)

py_test_simple(
    name = "cpp_tpl_types_test",
    size = "small",
    deps = [":cpp_types"],
)

cc_library(
    name = "cpp_template_cc",
    hdrs = [
        "cpp_template.h",
    ],
    deps = [
        ":cpp_template_types_cc",
        "//cpp:type_pack",
    ],
)

cc_library(
    name = "simple_converter_py",
    hdrs = [
        "simple_converter_py.h",
    ],
    deps = [
        ":cpp_template_cc",
        "//cpp:simple_converter",
    ],
)


pybind_library(
    name = "cpp_template_test_lib",
    py_deps = [
        ":cpp_template",
        ":cpp_types",
    ],
    cc_so_name = "test/_cpp_template_test",
    cc_srcs = [
        "test/_cpp_template_test.cc",
    ],
    cc_deps = [
        "//cpp:name_trait",
        ":cpp_template_cc",
    ],
)

py_test_simple(
    name = "cpp_template_test",
    size = "small",
    deps = [
        ":cpp_template_test_lib",
    ],
)

pybind_library(
    name = "scalar_type_test_lib",
    py_deps = [
        ":cpp_template",
        ":cpp_types",
    ],
    cc_so_name = "test/_scalar_type_test",
    cc_srcs = [
        "test/_scalar_type_test.cc",
    ],
    cc_deps = [
        ":cpp_template_cc",
        ":simple_converter_py",
        "//cpp:name_trait",
        "//cpp:simple_converter",
    ],
)

py_test_simple(
    name = "scalar_type_test",
    size = "small",
    deps = [":scalar_type_test_lib"],
)
