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

cc_binary(
    name = "test_keep_alive_transitive",
    srcs = [
         "custom_tests/test_keep_alive_transitive.cc",
         "tests/pybind11_tests.cpp",
    ],
    data = [
        "custom_tests/test_keep_alive_transitive.py",
    ],
    args = ["$(location custom_tests/test_keep_alive_transitive.py)"],
    deps = [":include"],
)

cc_binary(
    name = "test_alias_wrapper",
    srcs = [
         "custom_tests/test_alias_wrapper.cc",
         "tests/pybind11_tests.cpp",
    ],
    data = [
        "custom_tests/test_alias_wrapper.py",
    ],
    args = ["$(location custom_tests/test_alias_wrapper.py)"],
    deps = [":include"],
)

cc_binary(
    name = "test_const",
    srcs = [
         "custom_tests/test_const.cc",
         "tests/pybind11_tests.cpp",
    ],
    data = [
        "custom_tests/test_const.py",
    ],
    args = ["$(location custom_tests/test_const.py)"],
    deps = [":include"],
)

cc_binary(
    name = "test_unique_ptr_cast_error",
    srcs = [
         "custom_tests/test_unique_ptr_cast_error.cc",
         "tests/pybind11_tests.cpp",
    ],
    deps = [":include"],
)

cc_binary(
    name = "test_callback_eigen_ref",
    srcs = [
         "custom_tests/test_callback_eigen_ref.cc",
         "tests/pybind11_tests.cpp",
    ],
    deps = [":include"],
)

# Original tests.
cc_binary(
    name = "tests/pybind11_tests.so",
    srcs = [
        "tests/pybind11_tests.cpp",
#        "tests/test_numpy_dtype_user.cpp",
    ],
    linkstatic = 0,
    linkshared = 1,
    deps = [
        ":include",
        "@python//:python",
        "@numpy",
    ],
)

#py_library(
#    name = "tests",
#    imports = ["tests"],
#    data = [
#        "tests/pybind11_tests.so",
#    ],
#)

load(":orig_tests.bzl", "symlink")

symlink(
    name = "tests/libpybind11_tests.so",
    actual = "tests/pybind11_tests.so",
)

cc_binary(
    name = "tests/test_numpy_dtype_user",
    srcs = [
        "tests/test_numpy_dtype_user.cpp",
        "tests/pybind11_tests.cpp",
    ],
    data = ["tests/test_numpy_dtype_user.py"],
    deps = [
        ":include",
        "@python",
        "@numpy",
    ],
)
