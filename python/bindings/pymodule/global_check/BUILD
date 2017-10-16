# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module_share")

package(default_visibility = ["//visibility:private"])

cc_library(
  name = "producer",
  srcs = [
    "producer.h",
    "producer.cc",
  ],
  # If 0 - can simply load object code in. Yas.
  # If 1 - have to use `ShareSymbols` crap.
  linkstatic = 0,
)

pybind11_module_share(
    name = "consumer_1",
    # srcs = ["producer.h"],
    deps = [":producer"],
    package_dir = "../..",
)

pybind11_module_share(
    name = "consumer_2",
    # srcs = ["producer.h"],
    deps = [":producer"],
    package_dir = "../..",
)

py_library(
    name = "global_check",  # This is the package's name.
    deps = [
        ":consumer_1",
        ":consumer_2",
    ],
    visibility = ["//visibility:public"],
)

py_test_simple(
   name = "global_check_test",
   size = "small",
   deps = [":global_check"],
)

py_test_simple(
   name = "global_check_fail_test",
   size = "small",
   deps = [":global_check"],
)
