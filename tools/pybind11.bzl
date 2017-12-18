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
