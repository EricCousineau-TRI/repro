cc_library(
    name = "pybind11_numpy",
    deps = [
        "//python/pybind11:include",
        "@numpy//:numpy",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_rational.so",
    srcs = ["test_rational.cc"],
    linkshared = 1,
    linkstatic = 0,
    deps = [
        "//python/pybind11:include",
        "@python//:python",
        "@numpy",
    ],
)

cc_binary(
    name = "test_rational_min.so",
    srcs = ["test_rational_min.cc"],
    linkshared = 1,
    linkstatic = 0,
    deps = [
        "//python/pybind11:include",
        "@python//:python",
        "@numpy",
    ],
)

cc_binary(
    name = "test_basic",
    srcs = [
        "ufunc_utility.h",
        "ufunc_op.h",
        "test_basic.cc",
    ],
    deps = [
        ":pybind11_numpy",
        "//cpp:wrap_function",
        "@fmt",
    ],
    data = [
        "test_basic.py",
        "test_rational.so",
        # "test_rational_min.so",
    ],
)
