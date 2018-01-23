# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module", "pybind_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cpp_const",
    srcs = ["cpp_const.py"],
    imports = ["."],
)

py_test_simple(
    name = "cpp_const_test",
    deps = [":cpp_const"],
)

cc_library(
    name = "cpp_const_pybind",
    hdrs = ["cpp_const_pybind.h"],
    deps = [
        "//cpp:type_pack",
        "//cpp:wrap_func",
        "//python/pybind11:include",
    ],
)

pybind_library(
    name = "cpp_const_pybind_test_py",
    py_srcs = ["test/cpp_const_pybind_test.py"],
    py_deps = [
        ":cpp_const",
    ],
    cc_so_name = "test/_cpp_const_test_py",
    cc_srcs = [
        "test/_cpp_const_test.cc",
    ],
    cc_deps = [
        "//cpp:name_trait",
        ":cpp_const_pybind",
    ],
)

py_test_simple(
    name = "cpp_const_pybind_test",
    # main = "test/cpp_const_pybind_test.py",
    size = "small",
    deps = [
        ":cpp_const_pybind_test_py",
    ],
)
