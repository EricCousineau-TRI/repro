cc_library(
    name = "pybind11_numpy",
    deps = [
        "//python/pybind11:include",
        "@numpy//:numpy",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_basic",
    srcs = [
        "ufunc_utility.h",
        "test_basic.cc",
    ],
    deps = [
        ":pybind11_numpy",
    ],
    data = ["test_basic.py"],
)
