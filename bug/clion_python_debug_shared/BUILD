cc_binary(
    name = "noddy.so",
    srcs = ["noddy.c"],
    linkshared = 1,
    linkstatic = 0,
    deps = [
        "@python//:python",
    ],
)

py_library(
    name = "noddy",
    imports = ["."],
    data = [":noddy.so"],
)

py_test(
    name = "noddy_test",
    srcs = ["noddy_test.py"],
    deps = [":noddy"],
)
