# From an old version of Drake:

def _impl(repository_ctx):
    python_config = repository_ctx.which("python2.7-config")
    if not python_config:
        fail("bad")
    result = repository_ctx.execute([python_config, "--includes"])
    if result.return_code != 0:
        fail("bad")
    cflags = result.stdout.strip().split(" ")
    cflags = [cflag for cflag in cflags if cflag]
    root = repository_ctx.path("")
    root_len = len(str(root)) + 1
    base = root.get_child("include")
    includes = []
    for cflag in cflags:
        if cflag.startswith("-I"):
            source = repository_ctx.path(cflag[2:])
            destination = base.get_child(str(source).replace("/", "_"))
            include = str(destination)[root_len:]
            if include not in includes:
                repository_ctx.symlink(source, destination)
                includes += [include]
    result = repository_ctx.execute([python_config, "--ldflags"])
    if result.return_code != 0:
        fail("bad")
    linkopts = result.stdout.strip().split(" ")
    linkopts = [linkopt for linkopt in linkopts if linkopt]
    for i in reversed(range(len(linkopts))):
        if not linkopts[i].startswith("-"):
            linkopts[i - 1] += " " + linkopts.pop(i)
    prefix = repository_ctx.execute([python_config, "--prefix"]).stdout.strip()
    linkopts += ["-L{}/lib".format(prefix)]
    file_content = """
cc_library(
    name = "python",
    hdrs = glob(["include/**"]),
    includes = {},
    linkopts = {},
    visibility = ["//visibility:public"],
)
    """.format(includes, linkopts)
    repository_ctx.file("BUILD", content=file_content, executable=False)

python_repository = repository_rule(
    _impl,
    local = True,
)

def github_archive(
        name,
        repository = None,
        commit = None,
        sha256 = None,
        build_file = None,
        build_file_content = None,
        **kwargs):
    url = "https://github.com/%s/archive/%s.tar.gz" % (repository, commit)
    repository_split = repository.split("/")
    _, project = repository_split
    strip_commit = commit
    if commit[0] == 'v':
        # Github archives omit the "v" in version tags, for some reason.
        strip_commit = commit[1:]
    strip_prefix = project + "-" + strip_commit
    native.new_http_archive(
        name=name,
        urls=[url],
        sha256=sha256,
        build_file=build_file,
        build_file_content=build_file_content,
        strip_prefix=strip_prefix,
        **kwargs)

def pybind11_repository(name = "pybind11"):
    github_archive(
        name = name,
        repository = "pybind/pybind11",
        commit = "ce9d6e2c0d02019c957ad48dad86a06d54103565",
        sha256 = "7a9d0b99c2d6bc311a2261085bd991063226f512fe66ad4bd961cdb6653637c7",
        build_file_content = """
cc_library(
    name = "pybind11",
    hdrs = glob([
        "include/pybind11/**/*.h",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@python//:python",
    ],
)
""",
    )

def default_repositories():
    python_repository(name = "python")
    pybind11_repository(name = "pybind11")
