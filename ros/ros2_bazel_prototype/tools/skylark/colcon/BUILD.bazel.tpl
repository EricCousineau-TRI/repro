 # -*- python -*-

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cc",
    hdrs = glob(["{}/**/*".format(x) for x in %{cc_includes}]),
    includes = %{cc_includes},
    linkopts = %{cc_linkopts},
    deps = %{cc_deps},
)

py_library(
    name = "py",
    srcs = glob(["{}/**/*.py".format(x) for x in %{py_imports}]),
    data = glob(["{}/**/*.so".format(x) for x in %{py_imports}]),
    imports = %{py_imports},
    deps = %{py_deps},
)
