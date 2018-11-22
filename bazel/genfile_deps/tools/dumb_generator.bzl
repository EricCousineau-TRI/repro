USE_WORKAROUND = True
TOOL = "//tools:dumb_generator"

def dumb_generator(name, src, data):
    if USE_WORKAROUND:
        _dumb_generator_workaround(name, src, data)
    else:
        _dumb_generator_simple(name, src, data)

def _dumb_generator_simple(name, src, data):
    native.genrule(
        name = name + "_gen_simple",
        srcs = [src] + data,
        outs = [name],
        cmd = "{tool} --input {src} --output $@".format(
            tool = "$(location {})".format(TOOL),
            src = "$(location {})".format(src),
        ),
        tools = [TOOL],
    )

def _dumb_generator_workaround(name, src, data):
    wrap_tool = name + "_gen_tool"
    main = TOOL + ".py"
    data = [src] + data
    native.py_binary(
        name = wrap_tool,
        srcs = [main],
        main = main,
        data = data,
        deps = [TOOL],
    )
    native.genrule(
        name = name + "_gen_workaround",
        srcs = data,
        outs = [name],
        cmd = "{tool} --use_workaround --input {src} --output $@".format(
            tool = "$(location {})".format(wrap_tool),
            src = "$(location {})".format(src),
        ),
        tools = [wrap_tool],
    )
