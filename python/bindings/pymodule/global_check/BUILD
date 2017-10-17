# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load("//tools:python.bzl", "py_test_simple")
load("//tools:pybind11.bzl", "pybind11_module_share")

package(default_visibility = ["//visibility:private"])

# Check into drawbacks of: https://github.com/bazelbuild/bazel/issues/492
cc_library(
  name = "producer_linkstatic",
  hdrs = ["producer_linkstatic.h"],
  srcs = ["producer_linkstatic.cc"],
  linkstatic = 1,
)

cc_library(
  name = "producer_linkshared",
  hdrs = ["producer_linkshared.h"],
  srcs = ["producer_linkshared.cc"],
  # If 0 - can simply load object code in. Yas.
  # If 1 - have to use `ShareSymbols` crap.
  linkstatic = 0,
)

cc_library(
  name = "producer",
  hdrs = ["producer.h"],
  deps = [
    ":producer_linkshared",
    ":producer_linkstatic",
  ],
  linkstatic = 0,
)

cc_binary(
  name = "producer_main",
  srcs = ["producer_main.cc"],
  deps = [
    ":producer",
  ],
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
