def _impl(ctx):
    object_files = []
    for dep in ctx.attr.deps:
        # https://bazel.build/rules/lib/CcInfo
        info = dep[CcInfo]
        # https://bazel.build/rules/lib/LinkingContext
        for linker_input in info.linking_context.linker_inputs.to_list():
            for lib in linker_input.libraries:
                # https://bazel.build/rules/lib/LibraryToLink#pic_objects
                object_files += lib.pic_objects
        # for lib in info.linking_context:
        #     print(lib)
    return [
        DefaultInfo(files = depset(object_files))
    ]

extract_cc_object_files = rule(
    implementation = _impl,
    attrs = {
        "deps": attr.label_list(),
    },
)
