load("//tools/skylark:execute.bzl", "execute_or_fail")

_PROPERTIES = [
    "INCLUDE_DIRECTORIES",
    "LINK_FLAGS",
    "LINK_DIRECTORIES",
    "LINK_LIBRARIES",
]

_ISOLATE = r"""
#!/bin/bash
set -eux -o pipefail
env -i \
    HOME=$HOME \
    SHELL=$SHELL \
    USER=$USER \
    PATH=/usr/local/bin:/usr/bin:/bin \
    "$@"
""".lstrip()

_BUILD = r"""
#!/bin/bash
set -eux -o pipefail

{exports}

mkdir build && cd build
touch empty.cc
cmake .. "$@"

cd ..
mv build/props.txt .
rm -rf build
""".lstrip()

def _label(relpath):
    return Label("@ros2_bazel_prototype//tools/skylark/cmake:{}".format(relpath))

def _parse_cmake_props(text):
    props = dict()
    for line in text.strip().split("\n"):
        prop, value = line.split("=")
        props[prop] = value and value.split(";") or []
    return props

def _impl(repo_ctx):  # repository_ctx
    repo_ctx.template(
        "CMakeLists.txt",
        _label("cmake_cc.CMakeLists.txt.in"),
        substitutions = {
            "@NAME@": repo_ctx.name,
            "@PACKAGES@": " ".join(repo_ctx.attr.packages),
            "@PROPERTIES@": " ".join(_PROPERTIES),
        },
        executable = False,
    )
    env_vars = repo_ctx.attr.env_vars
    cache_entries = repo_ctx.attr.cache_entries
    exports = ["export {}={}".format(k, v) for k, v in env_vars.items()]
    repo_ctx.file("isolate.sh", content=_ISOLATE)
    repo_ctx.file("build.sh", content=_BUILD.format(exports="\n".join(exports)))
    # Configure.
    flags = ["-D{}={}".format(k, v) for k, v in cache_entries.items()]
    execute_or_fail(
        repo_ctx, ["./isolate.sh", "./build.sh"] + flags)
    props_text = execute_or_fail(repo_ctx, ["cat", "props.txt"]).stdout
    props = _parse_cmake_props(props_text)
    includes = []
    for include_path in props["INCLUDE_DIRECTORIES"]:
        include = "include/" + include_path.replace("/", "_")
        repo_ctx.symlink(repo_ctx.path(include_path), include)
        includes.append(include)
    linkopts = props["LINK_FLAGS"]
    linkopts += ["-L{}".format(x) for x in props["LINK_DIRECTORIES"]]
    for lib in props["LINK_LIBRARIES"]:
        index = lib.rfind("/")
        libdir = lib[:index]
        linkopts += ["-Wl,-rpath " + libdir, "-l{}".format(lib)]
    repo_ctx.template(
        "BUILD.bazel",
        _label("cmake_cc.BUILD.tpl"),
        substitutions = {
            "%{name}": repr(repo_ctx.name),
            "%{includes}": repr(includes),
            "%{linkopts}": repr(linkopts),
            "%{deps}": repr(repo_ctx.attr.deps),
        },
    )

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
