package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libexample_lib.so",
    srcs = [
        "example_lib.cc",
        "example_lib.h",
    ],
    linkshared = 1,
    linkstatic = 1,
)

cc_binary(
    name = "example_lib_py.so",
    srcs = [
        "example_lib_py.cc",
        "example_lib.h",
        "libexample_lib.so",
    ],
    deps = [
        "@pybind11",
    ],
    linkshared = 1,
    linkstatic = 1,
)

py_library(
    name = "example_lib_py",
    data = ["example_lib_py.so"],
    imports = ["."],
)

py_test(
    name = "example_lib_py_test",
    srcs = ["example_lib_py_test.py"],
    deps = [":example_lib_py"],
)
