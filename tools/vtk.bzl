# -*- python -*-
# See https://www.bazel.io/versions/master/docs/skylark/repository_rules.html

"""
Derived from @drake//tools/gurobi.bzl

In your shell, define:

    VTK_VERSION=vtk-5.10
    # Squirrel this away wherever you like.
    VTK_ROOT=~/.local/vtk/$VTK_VERSION

Build VTK as follows:

    cd drake-distro/externals/vtk
    mkdir -p build && cd build
    cmake .. -DCMAKE_CXX_FLAGS="-DGLX_GLXEXT_LEGACY=1" -DCMAKE_INSTALL_PREFIX=$VTK_ROOT -DBUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release
    make -j install

In your ~/.bash_aliases, use VTK_ROOT and VTK_VERSION and export hese:

    # Used by vtk_repository
    export VTK_INCLUDE=$VTK_ROOT/include/$VTK_VERSION
    export VTK_LIBDIR=$VTK_ROOT/lib/$VTK_VERSION
    # Necessary for execution
    export LD_LIBRARY_PATH=$VTK_LIBDIR:$LD_LIBRARY_PATH
"""

# TODO(eric.cousineau): Provide more granularity for libraries, such that you are not forced to link to ALL libraries.

def strip_indent(s, indent=4):
    # Must correct indentation for BUILD file at least.
    return s.replace("\n" + (" " * indent), "\n")  # Strip leading indent from lines.

def _vtk_impl(repository_ctx):
    vtk_include_path = repository_ctx.os.environ.get("VTK_INCLUDE", "")
    vtk_libdir_path = repository_ctx.os.environ.get("VTK_LIBDIR", "")
    vtk_include_sym = "vtk-system-inc"
    vtk_libdir_sym = "vtk-system-lib"

    repository_ctx.symlink(vtk_include_path or "/MISSING", vtk_include_sym)
    repository_ctx.symlink(vtk_libdir_path or "/MISSING", vtk_libdir_sym)

    if not vtk_include_path or not vtk_libdir_path:
        warning_detail = "VTK path is empty or unset"
    else:
        warning_detail = "VTK include (%s) / lib path ('%s') are invalid" % (vtk_include_path, vtk_libdir_path)
    warning = """

    WARNING: {detail}
    Please set VTK_INCLUDE and VTK_LIBDIR correctly

    """.format(detail=warning_detail)

    # # Cannot glob easily
    # print("Libs:\n%s" % native.glob(["%s/libvtk*.so" % vtk_libdir_path]))

    BUILD = """
    hdrs = glob(["{inc}/*.h"])
    libs = glob(["{libdir}/lib*.so"])

    print(""\"{warning}""\") \
        if not hdrs or not libs else \
        cc_library(
            name = "vtk",
            srcs = libs,
            hdrs = hdrs,
            linkstatic = 0,
            includes = ["{inc}"],
            visibility = ["//visibility:public"],
        )
    """.format(
        warning = strip_indent(warning),
        libdir=vtk_libdir_sym,
        inc=vtk_include_sym)
    repository_ctx.file(
        "BUILD",
        content=strip_indent(BUILD),
        executable=False)

vtk_repository = repository_rule(
    environ = ["VTK_INCLUDE", "VTK_LIBDIR"],
    local = True,
    implementation = _vtk_impl,
)

def vtk_test_tags():
    return ["vtk"]
