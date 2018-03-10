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
    srcs = ["test_basic.cc"],
    deps = [
        ":pybind11_numpy",
    ],
)
