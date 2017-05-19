# Testing to see if "tags" can be any iterable (e.g., a set)

# cur_tags = {"something"}
# new_tags = {"something", "else"}

sh_test(
    name = "basic_test",
    srcs = [
        "basic_test.sh",
    ],
    timeout = "short",
    tags = ["something", "else", "else"], # Cannot specify set, {}
)
