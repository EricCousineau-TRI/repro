# -*- python -*-

def symlink(
        name,
        actual = []):
    native.genrule(
        name = name,
        srcs = [actual],
        outs = [name],
        cmd = "ln -s $< $@",
        # tags = tags,
        # visibility = visibility,
    )
