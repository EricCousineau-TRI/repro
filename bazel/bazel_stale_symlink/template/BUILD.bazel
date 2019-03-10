genrule(
    name = "gen",
    srcs = ["input.txt"],
    outs = ["output.txt"],
    cmd = "cat $(location :input.txt) > $@",
)

py_test(
    name = "output_test",
    srcs = ["output_test.py"],
    data = [
        "input.txt",
        "output.txt",
    ],
)
