def _impl(ctx):
    object_files = []
    for dep in ctx.attr.deps:
        # https://bazel.build/rules/lib/CcInfo
        info = dep[CcInfo]
        # https://bazel.build/rules/lib/LinkingContext
        for linker_input in info.linking_context.linker_inputs.to_list():
            if not ctx.attr.transitive and linker_input.owner != dep.label:
                continue
            for lib in linker_input.libraries:
                print(lib)
                # https://bazel.build/rules/lib/LibraryToLink#pic_objects
                object_files += lib.pic_objects
    print(object_files)
    return [
        DefaultInfo(files = depset(object_files))
    ]

extract_cc_object_files = rule(
    implementation = _impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "transitive": attr.bool(default = False),
    },
)

def cc_shared_library(
        name,
        hdrs = [],
        srcs = [],
        deps = [],
        **kwargs):
    # WARNING: deps here is only into binary; other implementations should
    # forward headers and other stuff.
    # WARNING: Be careful to use `alwayslink = True` for deps coming in here.
    # Should use `whole_archive.bzl` instead.
    solib = "lib{}.so.1".format(name)
    native.cc_binary(
        name = solib,
        srcs = srcs + hdrs,
        linkshared = 1,
        linkstatic = 1,
        deps = deps,
        **kwargs
    )
    native.cc_library(
        name = name,
        hdrs = hdrs,
        srcs = [solib],
        **kwargs
    )
