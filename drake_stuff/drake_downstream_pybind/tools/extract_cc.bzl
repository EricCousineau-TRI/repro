# -*- python -*-
# vi: set ft=python :

"""
For extracting information from C++ targets.

See simple prototype and tests here:
https://github.com/EricCousineau-TRI/repro/tree/6025e1d5aa0/bazel/bazel_use_cc_precompile
"""  # noqa

def extract_cc_object_files_ctx(ctx, cc_info):
    # https://bazel.build/rules/lib/CcInfo
    object_files = []
    owners = []

    # https://bazel.build/rules/lib/LinkingContext
    for linker_input in cc_info.linking_context.linker_inputs.to_list():
        owner = linker_input.owner
        for lib in linker_input.libraries:
            # https://bazel.build/rules/lib/LibraryToLink#pic_objects
            object_files += lib.pic_objects
            owners += [owner] * len(lib.pic_objects)
    return depset(object_files).to_list(), owners

def _ensure_cc_object_files_nonempty_ctx(ctx, object_files):
    # Dunno why, but Bazel hates getting an empty list and trying to feed
    # that into cc_library(srcs). Workaround by adding dummy file if need be.
    # TODO(eric.cousineau): How to fix?
    if len(object_files) == 0:
        object_files, _ = extract_cc_object_files_ctx(
            ctx,
            cc_info = ctx.attr._empty_dep[CcInfo],
        )
    if len(object_files) == 0:
        fail("Unexpected")
    return object_files

_ensure_cc_object_files_nonempty_attrs = {
    "_empty_dep": attr.label(
        default = "@//tools:extract_cc_empty",
    ),
}

ExtractCcSrcs = provider(
    fields = [
        "srcs",
    ],
)

def _extract_cc_srcs_impl(target, ctx):
    srcs = getattr(ctx.rule.attr, "srcs", [])
    return [
        ExtractCcSrcs(
            srcs = srcs,
        ),
    ]

"""Extracts `srcs` Target lists for consumption by rules."""

extract_cc_srcs = aspect(
    implementation = _extract_cc_srcs_impl,
    attr_aspects = ["srcs"],
)

def extract_package_cc_hdrs_srcs_data(name, package_deps, reuse_object_files):
    """
    Extracts C++ hdrs, srcs, data in such a way that they can be reconsumed.

    This is intended for use with recompiling / relinking recompiling targets
    while replacing certain deps (e.g. going from static libraries to shared
    libraries for Python bindnigs).
    """

    hdrs_name = "_" + name + "_hdrs"
    _do_extract_package_cc_srcs_hdrs_data(
        name = hdrs_name,
        mode = "hdrs",
        package_deps = package_deps,
    )

    srcs_name = "_" + name + "_srcs"
    if reuse_object_files:
        srcs_mode = "srcs_object_files"
    else:
        srcs_mode = "srcs"
    _do_extract_package_cc_srcs_hdrs_data(
        name = srcs_name,
        mode = srcs_mode,
        package_deps = package_deps,
    )

    data_name = "_" + name + "_data"
    _do_extract_package_cc_srcs_hdrs_data(
        name = data_name,
        mode = "data",
        package_deps = package_deps,
    )

    # Include filegroup for tsting purposes.
    native.filegroup(
        name = name,
        testonly = 1,
        srcs = [
            hdrs_name,
            srcs_name,
            data_name,
        ],
    )

    return [hdrs_name], [srcs_name], [data_name]

def _do_extract_impl(ctx):
    mode = ctx.attr.mode
    if mode == "srcs":
        src_depsets = []
        for dep in ctx.attr.package_deps:
            for src in dep[ExtractCcSrcs].srcs:
                src_depsets.append(src.files)
        return DefaultInfo(files = depset(transitive = src_depsets))
    elif mode == "srcs_object_files":
        object_files = []
        for dep in ctx.attr.package_deps:
            new_object_files, owners = extract_cc_object_files_ctx(
                ctx,
                cc_info = dep[CcInfo],
            )
            for object_file, owner in zip(new_object_files, owners):
                # Direct dependencies.
                if owner == dep.label:
                    object_files.append(object_file)
        object_files = _ensure_cc_object_files_nonempty_ctx(ctx, object_files)
        return DefaultInfo(files = depset(object_files))
    elif mode == "hdrs":
        hdrs = []
        for dep in ctx.attr.package_deps:
            hdrs += dep[CcInfo].compilation_context.direct_public_headers
        return DefaultInfo(files = depset(hdrs))
    elif mode == "data":
        data_runfiles = []
        for dep in ctx.attr.package_deps:
            data_runfiles.append(dep[DefaultInfo].data_runfiles)
        runfiles = ctx.runfiles().merge_all(data_runfiles)
        return DefaultInfo(runfiles = runfiles)
    else:
        fail("Bad mode")

_attrs = {
    "mode": attr.string(
        values = ["hdrs", "srcs", "srcs_object_files", "data"],
    ),
    "package_deps": attr.label_list(
        providers = [CcInfo, DefaultInfo],
        aspects = [extract_cc_srcs],
    ),
}
_attrs.update(_ensure_cc_object_files_nonempty_attrs)

_do_extract_package_cc_srcs_hdrs_data = rule(
    implementation = _do_extract_impl,
    attrs = _attrs,
)
