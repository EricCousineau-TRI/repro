 # -*- python -*-

package(default_visibility = ["//visibility:public"])

cc_library(
    name = %{name},
    hdrs = glob(["{}/**/*".format(x) for x in %{includes}]),
    includes = %{includes},
    linkopts = %{linkopts},
    deps = %{deps},
)
