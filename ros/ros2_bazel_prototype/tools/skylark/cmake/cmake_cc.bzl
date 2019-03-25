load("//tools/skylark:execute.bzl", "execute_or_fail")

def _label(relpath):
    return Label("@ros2_bazel_prototype//tools/skylark/cmake:{}".format(relpath))

def _impl(repo_ctx):
    repo_ctx.symlink(repo_ctx.attr.template_cmakelists, "CMakeLists.txt.in")
    repo_ctx.symlink(_label("cc_BUILD.tpl"), "BUILD.tpl")
    config = dict(
        name=repo_ctx.name,
        packages=repo_ctx.attr.packages,
        env_vars=repo_ctx.attr.env_vars,
        cache_entries=repo_ctx.attr.cache_entries,
        deps=repo_ctx.attr.deps,
        libdir_order_preference=repo_ctx.attr.libdir_order_preference,
    )
    # For optional overlays :(
    for prefix, archive in repo_ctx.attr.archives.items():
        repo_ctx.download_and_extract(archive, output=prefix)
    repo_ctx.template(
        "build.py",
        _label("cc_build.py.tpl"),
        substitutions = {
            "%{config}": repr(config),
        },
    )
    execute_or_fail(repo_ctx, ["./build.py"], quiet=False)

"""
Extracts relevant properties from CMake stuff.

Unfortunately, I (Eric) dunno how to make `rules_foreign_cc` tell more about
the transitive deps, like headers and other libs. Geared towards usage with
`ament` (ROS2).

Borrows some args from `cmake_external` (rules_foreign_cc).
"""

cmake_cc_repository = repository_rule(
    attrs = dict(
        # TODO(eric): Add licenses in the package list?
        packages = attr.string_list(mandatory = True),
        cache_entries = attr.string_dict(),
        env_vars = attr.string_dict(),
        deps = attr.label_list(),
        template_cmakelists = attr.label(
            default = _label("cc_ament_CMakeLists.txt.in")),
        # For hacking with overlays...
        libdir_order_preference = attr.string_list(),
        archives = attr.string_dict(),
    ),
    local = True,
    implementation = _impl,
)
