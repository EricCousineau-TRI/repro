# -*- python -*-

cc_library(
    name = "pybind11",
    hdrs = glob(["include/pybind11/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen//:eigen",
        "@numpy//:numpy",
        "@python//:python",
    ],
)
