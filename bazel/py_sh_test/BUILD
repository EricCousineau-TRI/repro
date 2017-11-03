py_binary(
    name = "single_file",
    srcs = ["single_file.py"],
)

py_binary(
    name = "multi_file",
    srcs = ["multi_file.py", "single_file.py"],
)

sh_test(
    name = "single_file_test",
    srcs = [":single_file"],
)

cc_binary(
    name = "single_file_cc",
    srcs = ["single_file.cc"],
)

sh_test(
    name = "single_file_cc_test",
    srcs = ["single_file_cc"],
)

