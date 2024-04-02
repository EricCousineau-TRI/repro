from bazel_tools.tools.python.runfiles.runfiles import Create as _Create

_runfiles = _Create()


def Rlocation(respath):
    result = _runfiles.Rlocation(respath)
    if result is None:
        raise RuntimeError(
            f"Resource path {respath} could not be resolved to a "
            "filesystem path; maybe there is a typo in the path, or a "
            "missing data = [] attribute in the BUILD.bazel file.")
    return result


def SubstituteMakeVariableLocation(arg):
    """Given a string argument that might be a $(location //foo) substitution,
    looks up ands return the specified runfile location for $(location //foo)
    if the argument is in such a form, or if not just returns the argument
    unchanged.  Only absolute labels ("//foo" or "@drake//bar") are supported.
    It is an error if the argument looks any other $(...).  For details see
    https://docs.bazel.build/versions/master/be/make-variables.html.
    """
    if arg.startswith("$(location "):
        label = arg[11:-1]
        assert label.startswith("@") or label.startswith("//"), label
        if not label.startswith("@"):
            label = "@anzu" + label
        elif label.startswith("@//"):
            label = label.replace("@//", "@anzu//")
        normalized = label[1:]  # Strip the leading @.
        normalized = normalized.replace("//:", "/")
        normalized = normalized.replace("//", "/")
        normalized = normalized.replace(":", "/")
        arg = Rlocation(normalized)
    assert not arg.startswith("$("), arg
    return arg
