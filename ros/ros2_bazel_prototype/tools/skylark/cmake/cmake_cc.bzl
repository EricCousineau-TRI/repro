load("//tools/skylark:execute.bzl", "execute_or_fail")

def _label(relpath):
    return Label("@ros2_bazel_prototype//tools/skylark/cmake:{}".format(relpath))

def _impl(repo_ctx):  # repository_ctx
    repo_ctx.symlink(_label("cmake_cc.CMakeLists.txt.in"), "CMakeLists.txt.in")
    repo_ctx.symlink(_label("cmake_cc.BUILD.tpl"), "BUILD.tpl")
    config = dict(
        name=repo_ctx.name,
        packages=repo_ctx.attr.packages,
        env_vars=repo_ctx.attr.env_vars,
        cache_entries=repo_ctx.attr.cache_entries,
        deps=repo_ctx.attr.deps,
    )
    repo_ctx.template(
        "build.py",
        _label("cmake_cc_setup.py.tpl"),
        substitutions = {
            "%{config}": repr(config),
        },
    )
    execute_or_fail(repo_ctx, ["./build.py"], quiet=False)

cmake_cc_repository = repository_rule(
    attrs = dict(
        # TODO(eric): Add licenses in the package list?
        packages = attr.string_list(mandatory = True),
        cache_entries = attr.string_dict(),
        env_vars = attr.string_dict(),
        deps = attr.label_list(),
    ),
    local = True,
    implementation = _impl,
)
