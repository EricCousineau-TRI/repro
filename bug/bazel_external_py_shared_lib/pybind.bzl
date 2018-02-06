def _python_repository(repository_ctx):
    python_config = repository_ctx.which("python{}-config".format(
        repository_ctx.attr.version))
    result = repository_ctx.execute([python_config, "--includes"])
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
    linkopts = result.stdout.strip().split(" ")
    linkopts = [linkopt for linkopt in linkopts if linkopt]
    for i in reversed(range(len(linkopts))):
        if not linkopts[i].startswith("-"):
            linkopts[i - 1] += " " + linkopts.pop(i)
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
    _python_repository,
    attrs = {"version": attr.string(default = "2.7")},
    local = True,
)

def pybind11_repository(name):
    commit = "add56ccdcac23a6c522a2c1174a866e293c61dab"
    sha256 = "d6cc302d0fcf508d80759934c1b40651df1278882998088c4e78dc175e471f52"
    project = "pybind11"
    repository = "pybind/pybind11"
    url = "https://github.com/%s/archive/%s.tar.gz" % (repository, commit)
    native.new_http_archive(
        name = name,
        urls = [url],
        strip_prefix = project + "-" + commit,
        sha256 = sha256,
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

def pybind_library(
        name,
        cc_srcs = [],
        cc_deps = [],
        py_srcs = [],
        py_deps = [],
        py_imports = [],
        visibility = None,
        testonly = None):
    py_name = name
    cc_so_name = name + ".so"
    # Add C++ shared library.
    native.cc_binary(
        name = cc_so_name,
        srcs = cc_srcs,
        # This is how you tell Bazel to create a shared library.
        linkshared = 1,
        linkstatic = 1,
        deps = [
            "@pybind11",
        ] + cc_deps,
        testonly = testonly,
        visibility = visibility,
    )
    # Add Python library.
    native.py_library(
        name = py_name,
        data = [cc_so_name],
        srcs = py_srcs,
        deps = py_deps,
        imports = py_imports,
        testonly = testonly,
        visibility = visibility,
    )
