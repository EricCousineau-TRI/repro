config_setting(
    name = "has_1",
    define_values = { "HAS_1": "ON" },
)

config_setting(
    name = "has_2",
    define_values = { "HAS_2": "ON" },
)

sh_binary(
    name = "script",
    srcs = ["script.sh"],
    args = select({
        ":has_1": ["has_1"],
        "//conditions:default": [],
    }) + select({
        ":has_2": ["has_2"],
        "//conditions:default": [],
    }),
)
