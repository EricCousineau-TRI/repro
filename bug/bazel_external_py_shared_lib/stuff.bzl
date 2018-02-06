def cc_shared_library(
        name,
        solib_name = None,
        hdrs = None,
        srcs = None,
        linkstatic = 0,
        linkshared = 1,
        linkopts = [],
        deps = None,
        # cc_library flags
        include_prefix = None,
        strip_include_prefix = None,
        includes = None,
        # Other flags.
        **kwargs):
    """Declares shared library that carries its dependencies (headers and other
    shared libraries) transitively through Bazel.
    """
    if linkshared != 1:
        fail("`cc_solib_library` only be used with `linkshared = 1`.")
    if solib_name == None:
        solib_name = "lib{}.so".format(name)
    hdrlib = name + ".headers"
    solib = name + ".solib"
    # Headers and upstream dependencies (for transitive consumption).
    native.cc_library(
        name = hdrlib,
        hdrs = hdrs,
        deps = deps,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        includes = includes,
        **kwargs)
    # Shared library artifact.
    native.cc_binary(
        name = solib_name,
        srcs = srcs,
        linkshared = 1,
        linkstatic = linkstatic,
        linkopts = linkopts,
        deps = [hdrlib],
        **kwargs)
    # Alias library naming for consistent consumption.
    native.alias(
        name = solib,
        actual = solib_name,
    )
    # Development glue.
    native.cc_library(
        name = name,
        srcs = [solib],
        deps = [hdrlib],
        linkstatic = 1,
        **kwargs)
    return (hdrlib, solib)
