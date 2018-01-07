cc_library(
    name = "include",
    srcs = glob([
        "include/**/*.h",
        "tests/*.h",
    ]),
    includes = ["include", "tests"],
    deps = [
        "@eigen",
        # "@numpy//:numpy",
        "@python",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "include3",
    srcs = glob([
        "include/**/*.h",
        "tests/*.h",
    ]),
    includes = ["include", "tests"],
    deps = [
        "@eigen",
        "@python3//:python",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_unique_ptr_cast",
    srcs = [
        "custom_tests/test_unique_ptr_cast.cc",
    ],
    deps = [
        ":include",
    ],
)

cc_binary(
    name = "test_shared_ptr_owner",
    srcs = [
        "custom_tests/test_shared_ptr_owner.cc",
    ],
    deps = [
        ":include",
    ],
)

cc_binary(
    name = "test_shared_ptr_owner3",
    srcs = [
        "custom_tests/test_shared_ptr_owner.cc",
    ],
    deps = [
        ":include3",
    ],
)

cc_binary(
    name = "test_ref_count",
    srcs = [
        "custom_tests/test_ref_count.cc",
    ],
    deps = [
        ":include",
    ],
)

cc_binary(
    name = "test_poly",
    srcs = ["custom_tests/test_poly.cc"],
    deps = [":include"],
)

cc_binary(
    name = "test_unique_ptr_stuff",
    srcs = ["custom_tests/test_unique_ptr_stuff.cc"],
    deps = [":include"],
)

#exports_files(["custom_tests/test_unique_ptr_keep_alive.py"])

# Debuggable things.

cc_binary(
    name = "test_unique_ptr",
    srcs = [
         "custom_tests/test_unique_ptr.cc",
         "tests/pybind11_tests.cpp",
    ],
    data = [
        "custom_tests/test_unique_ptr.py",
    ],
    args = ["$(location custom_tests/test_unique_ptr.py)"],
    deps = [":include"],
)

cc_binary(
    name = "test_unique_ptr_keep_alive",
    srcs = [
         "custom_tests/test_unique_ptr_keep_alive.cc",
         "tests/pybind11_tests.cpp",
    ],
    data = [
        "custom_tests/test_unique_ptr_keep_alive.py",
    ],
    args = ["$(location custom_tests/test_unique_ptr_keep_alive.py)"],
    deps = [":include"],
)

cc_binary(
    name = "test_callback_lvalue",
    srcs = [
         "custom_tests/test_callback_lvalue.cc",
         "tests/pybind11_tests.cpp",
    ],
    data = [
        "custom_tests/test_callback_lvalue.py",
    ],
    args = ["$(location custom_tests/test_callback_lvalue.py)"],
    deps = [":include"],
)
