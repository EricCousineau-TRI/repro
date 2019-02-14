from bazel_py_example.mid.library1 import library1_func


def library2_func():
    return library1_func() + 1
