load(
    "@rules_python//python:defs.bzl",
    "py_library",
)

def pybind_py_library(
        name,
        cc_srcs = [],
        cc_deps = [],
        cc_copts = [],
        cc_so_name = None,
        cc_binary_rule = native.cc_binary,
        py_srcs = [],
        py_deps = [],
        py_imports = [],
        py_library_rule = py_library,
        **kwargs):
    """Declares a pybind11 Python library with C++ and Python portions.

    @param cc_srcs
        C++ source files.
    @param cc_deps (optional)
        C++ dependencies.
        At present, these should be libraries that will not cause ODR
        conflicts (generally, header-only).
    @param cc_so_name (optional)
        Shared object name. By default, this is `${name}`, so that the C++
        code can be then imported in a more controlled fashion in Python.
        If overridden, this could be the public interface exposed to the user.
    @param py_srcs (optional)
        Python sources.
    @param py_deps (optional)
        Python dependencies.
    @param py_imports (optional)
        Additional Python import directories.
    @return struct(cc_so_target = ..., py_target = ...)
    """
    py_name = name
    if not cc_so_name:
        cc_so_name = name

    # TODO(eric.cousineau): See if we can keep non-`*.so` target name, but
    # output a *.so, so that the target name is similar to what is provided.
    cc_so_target = cc_so_name + ".so"

    # GCC and Clang don't always agree / succeed when inferring storage
    # duration (#9600). Workaround it for now.
    copts = cc_copts

    # Add C++ shared library.
    cc_binary_rule(
        name = cc_so_target,
        srcs = cc_srcs,
        # This is how you tell Bazel to create a shared library.
        linkshared = 1,
        linkstatic = 1,
        copts = copts,
        # Always link to pybind11.
        deps = [
            "@pybind11",
        ] + cc_deps,
        **kwargs
    )

    # Add Python library.
    py_library_rule(
        name = py_name,
        data = [cc_so_target],
        srcs = py_srcs,
        deps = py_deps,
        imports = py_imports,
        **kwargs
    )
    return struct(
        cc_so_target = cc_so_target,
        py_target = py_name,
    )
