load(":jupyter.bzl",
    "py_jupyter_binary",
    "py_jupyter_test",
)

py_library(
    name = "my_library",
    srcs = ["src/my_library.py"],
    imports = ["src"],
)

py_jupyter_binary(
    name = "simple_notebook",
    deps = [":my_library"],
)

py_jupyter_binary(
    name = "subdir/simple_notebook",
    deps = [":my_library"],
)

py_jupyter_test(
    name = "simple_notebook_test",
    deps = [":my_library"],
)
