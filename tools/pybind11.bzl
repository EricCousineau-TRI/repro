load("//tools:python.bzl", "python_repository")

copts = [
    "-Wno-#warnings",
    "-Wno-cpp",
    "-Wno-unknown-warning-option",
    ]

def pybind11_module(name, srcs = [], cc_deps = [], py_deps = [], package_dir = "..", imports = [], **kwargs):
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
        deps = cc_deps + [
            "@pybind11//:pybind11",
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

def drake_py_test(name, srcs = [], **kwargs):
    py_main = "test/{}.py".format(name)
    native.py_test(
        name = name,
        srcs = [
            py_main,
            ] + srcs,
        main = py_main,
        **kwargs
    )
