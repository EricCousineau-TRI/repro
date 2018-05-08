package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libupstream.so",
    srcs = ["upstream.cc"],
    linkshared = 1,
    linkstatic = 1,
)

cc_library(
    name = "upstream",
    srcs = ["libupstream.so"],
)

cc_binary(
    name = "libexample_lib.so",
    srcs = [
        "example_lib.cc",
        "example_lib.h",
    ],
    deps = [":upstream"],
    linkshared = 1,
    linkstatic = 1,
)

cc_import(
    name = "example_lib",
    shared_library = "libexample_lib.so",
)

cc_binary(
    name = "example_lib_py.so",
    srcs = [
        "example_lib_py.cc",
        "example_lib.h",
    ],
    deps = [
        # Cannot do transitives.
        ":example_lib",
        ":upstream",
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

py_library(
    name = "sub_level",
    deps = [
        ":example_lib_py",
    ],
)

py_test(
    name = "example_lib_py_test",
    srcs = ["example_lib_py_test.py"],
    deps = [":example_lib_py"],
)
