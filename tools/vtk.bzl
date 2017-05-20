# -*- python -*-
# See https://www.bazel.io/versions/master/docs/skylark/repository_rules.html

# Derived from @drake//tools/gurobi.bzl

def _vtk_impl(repository_ctx):
    vtk_include_path = repository_ctx.os.environ.get("VTK_INCLUDE", "")
    vtk_libdir_path = repository_ctx.os.environ.get("VTK_LIBDIR", "")
    vtk_include_sym = "vtk-system-inc"
    vtk_libdir_sym = "vtk-system-lib"

    repository_ctx.symlink(vtk_include_path or "/MISSING", vtk_include_sym)
    repository_ctx.symlink(vtk_libdir_path or "/MISSING", vtk_libdir_sym)

    if not vtk_include_path:
        warning_detail = "VTK path is empty or unset"
    else:
        warning_detail = "VTK include / lib path '%s' is invalid" % vtk_include_path
    warning = warning_detail

    # # Cannot glob easily
    # print("Libs:\n%s" % native.glob(["%s/libvtk*.so" % vtk_libdir_path]))

    BUILD = """
hdrs = glob(["{inc}/*.h"])
libs = glob(["{libdir}/lib*.so"])

print(warning) \
    if not hdrs or not libs else \
    cc_library(
        name = "vtk",
        srcs = libs,
        hdrs = hdrs,
        linkstatic = 0,
        includes = ["{inc}"],
        visibility = ["//visibility:public"],
    )
""".format(warning=warning, libdir=vtk_libdir_sym, inc=vtk_include_sym)
    repository_ctx.file("BUILD", content=BUILD, executable=False)

vtk_repository = repository_rule(
    environ = ["VTK_INCLUDE", "VTK_LIBDIR"],
    local = True,
    implementation = _vtk_impl,
)
