load("//tools/skylark:execute.bzl", "execute_or_fail")

def _label(relpath):
    return Label("@ros2_bazel_prototype//tools/skylark/colcon:" + relpath)

def _impl(repo_ctx):
    repo_ctx.symlink(_label("cc_ament_CMakeLists.txt.in"), "CMakeLists.txt.in")
    repo_ctx.symlink(_label("BUILD.bazel.tpl"), "BUILD.bazel.tpl")
    config = dict(
        name=repo_ctx.name,
        workspaces=repo_ctx.attr.workspaces,
        cc_packages=repo_ctx.attr.cc_packages,
        cc_cache_entries=repo_ctx.attr.cc_cache_entries,
        cc_deps=repo_ctx.attr.cc_deps,
        py_packages=repo_ctx.attr.py_packages,
        py_deps=repo_ctx.attr.py_deps,
    )
    # For optional overlays :(
    for prefix, archive in repo_ctx.attr.overlay_archives.items():
        repo_ctx.download_and_extract(archive, output=prefix)
    repo_ctx.template(
        "build.py",
        _label("build.py.tpl"),
        substitutions = {
            "%{config}": repr(config),
        },
    )
    execute_or_fail(repo_ctx, ["./build.py"], quiet=False)

"""
Extracts relevant properties from CMake and Python stuff for colcon install
trees. Uses `ament` for C++ (CMake), and generic Python stuff for Python.
"""

# Unfortunately, I (Eric) dunno how to make `rules_foreign_cc` tell more about
# the transitive deps, like headers and other libs. Geared towards usage with
# `ament` (ROS2).

# TODO(eric): How to granularize dependency grouping? (e.g. only take msggen
# bits for translators, but take all RMW for other stuff?)
# TODO(eric): How to handle licenses?
colcon_repository = repository_rule(
    attrs = dict(
        # Workspaces
        # - FHS install trees (incl. ABI compatible overlays).
        workspaces = attr.string_list(),
        # - For distributing ABI compatbile overlays. Format is
        # `{local_path: archive}`.
        overlay_archives = attr.string_dict(),
        # C++
        cc_packages = attr.string_list(mandatory = True),
        # - N.B. `CMAKE_PREFIX_PATH` is already handled from `workspaces`.
        cc_cache_entries = attr.string_dict(),
        cc_deps = attr.label_list(),
        # Python
        py_packages = attr.string_list(mandatory = True),
        py_deps = attr.label_list(),
    ),
    local = True,
    implementation = _impl,
)
