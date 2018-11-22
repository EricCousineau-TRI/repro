def dumb_generator(name, src, data):
    tool = "//tools:dumb_generator"
    native.genrule(
        name = name + "_simple_gen",
        srcs = [src] + data,
        outs = [name],
        cmd = "$(location {tool}) --input $(location {src}) --output $@".format(
            tool = tool,
            src = src,
        ),
        tools = ["//tools:dumb_generator"],
    )
