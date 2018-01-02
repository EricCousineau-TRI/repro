load("//tools:python.bzl", "python_repository")

copts = [
    "-Wno-#warnings",
    "-Wno-cpp",
    "-Wno-unknown-warning-option",
    ]

def pybind11_binary(name, srcs, pybind11_deps = []):
    pass
    # native.cc_binary(
    #     name = name,
    #     srcs = srcs,
    #     deps = deps,
    # )

def pybind11_module(name, srcs = [], deps = [], py_deps = [], package_dir = "..", imports = [], **kwargs):
    cc_lib = "_{}".format(name)
    cc_lib_so = "_{}.so".format(name)
    native.cc_binary(
        name = cc_lib_so,
        srcs = srcs + [
            "{}.cc".format(cc_lib),
            ],
        copts = copts,
        linkshared = 1,
        linkstatic = 0,
        deps = deps + [
            # "@pybind11//:pybind11",
            "//python/pybind11:include",
        ],
    )

    native.py_library(
        name = name,
        srcs = [
            "__init__.py",
            "{}.py".format(name),
        ],
        data = [
            ":{}".format(cc_lib_so),
        ],
        deps = py_deps,
        imports = imports + [package_dir],
        visibility = ["//visibility:public"],
        **kwargs
    )

"""
Add ...share_symbols to enable RTLD_GLOBAL when importing SO files.
"""
def pybind11_module_share(name, py_deps = [], **kwargs):
    pybind11_module(
        name = name,
        py_deps = py_deps + [
            "//python/bindings/pymodule/util:share_symbols",
        ],
        **kwargs
    )

# From drake:
_BASE_PACKAGE = "python/bindings"

def _get_child_library_info(package = None, base_package = _BASE_PACKAGE):
    # Gets a package's path relative to a base package, and the sub-package
    # name (for installation).
    # @return struct(rel_path, sub_package)
    if package == None:
        package = native.package_name()
    base_package_pre = base_package + "/"
    if not package.startswith(base_package_pre):
        fail("Invalid package '{}' (not a child of '{}')"
             .format(package, base_package))
    sub_package = package[len(base_package_pre):]
    # Count the number of pieces.
    num_pieces = len(sub_package.split("/"))
    # Make the number of parent directories.
    rel_path = "/".join([".."] * num_pieces)
    return struct(rel_path = rel_path, sub_package = sub_package)

def _pybind_cc_binary(
        name,
        srcs = [],
        deps = [],
        visibility = None,
        testonly = None):
    """Declares a pybind11 shared library.

    The defines the library with the given name and srcs.
    The libdrake.so library and its headers are already automatically
    depended-on by this rule.
    """
    # TODO(eric.cousineau): Ensure `deps` is header-only, if this code is to
    # live longer.
    native.cc_binary(
        name = name,
        srcs = srcs,
        # This is how you tell Bazel to create a shared library.
        linkshared = 1,
        linkstatic = 1,
        # For all pydrake_foo.so, always link to Drake and pybind11.
        deps = [
            "//python/pybind11:include",
        ] + deps,
        testonly = testonly,
        visibility = visibility,
    )

def pybind_library(
        name,
        cc_srcs = [],
        cc_deps = [],
        cc_so_name = None,
        py_srcs = [],
        py_deps = [],
        py_imports = [],
        visibility = None,
        testonly = None):
    """Declares a pybind11 library with C++ and Python portions.

    @param cc_srcs
        C++ source files.
    @param cc_deps (optional)
        C++ dependencies.
        At present, these should be header only, as they will violate ODR
        with statically-linked libraries.
    @param cc_so_name (optional)
        Shared object name. By default, this is `_${name}`, so that the C++
        code can be then imported in a more controlled fashion in Python.
        If overridden, this could be the public interface exposed to the user.
    @param py_srcs
        Python sources.
    @param py_deps
        Python dependencies.
    @param py_imports
        Additional Python import directories.
    """
    py_name = name
    if not cc_so_name:
        cc_so_name = "_" + name
    # TODO(eric.cousineau): See if we can keep non-`*.so` target name, but
    # output a *.so, so that the target name is similar to what is provided.
    cc_so_name += ".so"
    # Add C++ shared library.
    _pybind_cc_binary(
        name = cc_so_name,
        srcs = cc_srcs,
        deps = cc_deps,
        testonly = testonly,
        visibility = visibility,
    )
    # Get current package's information.
    library_info = _get_child_library_info()
    # Add Python library.
    native.py_library(
        name = py_name,
        data = [cc_so_name],
        srcs = py_srcs,
        deps = py_deps,
        imports = [library_info.rel_path] + py_imports,
        testonly = testonly,
        visibility = visibility,
    )
