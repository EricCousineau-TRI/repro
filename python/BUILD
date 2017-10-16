package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11_include",
    srcs = glob([
        "pybind11_include/**/*.h",
    ]),
    includes = ["pybind11_include"],
    deps = [
        "@eigen//:eigen",
        # "@numpy//:numpy",
        "@python//:python",
    ],
)
