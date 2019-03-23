 # -*- python -*-

package(default_visibility = ["//visibility:public"])

cc_library(
    name = %{name},
    includes = %{includes},
    linkopts = %{linkopts},
    deps = %{deps},
)
