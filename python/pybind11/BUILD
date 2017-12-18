cc_library(
    name = "include",
    srcs = glob([
        "include/**/*.h",
    ]),
    includes = ["include"],
    deps = [
        "@eigen//:eigen",
        # "@numpy//:numpy",
        "@python//:python",
    ],
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
