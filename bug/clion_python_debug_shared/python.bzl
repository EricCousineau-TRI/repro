# -*- mode: python -*-
# vi: set ft=python :

# Derived from RobotLocomotion/drake's Pyton rules.

def _impl(repository_ctx):
    python_config = repository_ctx.which("python{}-config".format(
        repository_ctx.attr.version))

    if not python_config:
        fail("Could NOT find python{}-config".format(
            repository_ctx.attr.version))

    result = repository_ctx.execute([python_config, "--includes"])

    if result.return_code != 0:
        fail("Could NOT determine Python includes", attr=result.stderr)

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
        fail("Could NOT determine Python linkopts", attr=result.stderr)

    linkopts = result.stdout.strip().split(" ")
    linkopts = [linkopt for linkopt in linkopts if linkopt]

    print("Python: {}".format(repository_ctx.attr.name))
    print("  Config: {}".format(python_config))
    print("  Cflags: {}".format(cflags))
    print("  Include: {}".format(includes))
    print("  Libs: {}".format(linkopts))

    for i in reversed(range(len(linkopts))):
        if not linkopts[i].startswith("-"):
            linkopts[i - 1] += " " + linkopts.pop(i)

    prefix = repository_ctx.execute([python_config, "--prefix"]).stdout.strip()

    # If Python is compiled with `--enable-shared`.
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
    attrs = {"version": attr.string(default = "2.7")},
    local = True,
)
