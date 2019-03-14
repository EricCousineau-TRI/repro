py_library(
    name = "sub_example",
    srcs = [
        "__init__.py",
        "module.py",
    ],
    imports = [".."],
    visibility = ["//visibility:public"],
)

py_test(
    name = "usage_test",
    srcs = ["test/usage_test.py"],
    deps = [
        ":sub_example",
    ],
)
