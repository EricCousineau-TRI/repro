# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "_dup.so",
    srcs = [
        "_dup.cc",
    ],
    copts = [
        "-Wno-#warnings",
        "-Wno-cpp",
        "-Wno-unknown-warning-option",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "@pybind11//:pybind11",
    ],
)

cur_package = "//python/pybind11/dup_check/"

py_library(
    name = "dup_check",
    srcs = [
        "__init__.py",
    ],
    data = [
        ":_dup.so",
        # Better way to relative paths?
        cur_package + "sub", 
    ],
    imports = [".."],
)

py_test(
   name = "dup_check_test",
   size = "small",
   srcs = ["test/testDup.py"],
   main = "test/testDup.py",
   tags = [],
   deps = [":dup_check"],
)
