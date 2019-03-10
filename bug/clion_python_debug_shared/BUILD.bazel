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

# In debug mode, this does not (a) find the correct Python interpreter, and (b)
# does not find the correct directory.
cc_binary(
    name = "from_cc",
    srcs = ["from_cc.cc"],
    data = [":noddy_test"],
    testonly = 1,
)
