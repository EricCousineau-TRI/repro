# https://docs.bazel.build/versions/master/skylark/cookbook.html#runfiles-and-location-substitution

def _impl(ctx):
    executable = ctx.outputs.executable
    command = ctx.attr.command
    # Expand the label in the command string to a runfiles-relative path.
    # The second arg is the list of labels that may be expanded.
    print(ctx.attr.data)
    print(dir(ctx.attr.data))
    f = ctx.attr.data[1]
    print(f)
    print(dir(f))
    print(f.data_runfiles.files)
    command = ctx.expand_location(command, ctx.attr.data)
    # Create the output executable file with command as its content.
    ctx.file_action(
        output=executable,
        content=command,
        executable=True)

    return [DefaultInfo(
        # Create runfiles from the files specified in the data attribute.
        # The shell executable - the output of this rule - can use them at
        #    runtime. It is also possible to define data_runfiles and
        # default_runfiles. However if runfiles is specified it's not possible to
        # define the above ones since runfiles sets them both.
        # Remember, that the struct returned by the implementation function needs
        # to have a field named "runfiles" in order to create the actual runfiles
        # symlink tree.
        runfiles=ctx.runfiles(files=ctx.files.data)
    )]

execute = rule(
    implementation=_impl,
    executable=True,
    attrs={
        "command": attr.string(),
        "data": attr.label_list(cfg="data", allow_files=True),
    },
)

def _recursive_filegroup_impl(ctx):
    files = depset()
    for d in ctx.attr.data:
        files += d.data_runfiles.files
    if ctx.attr.dummy and not files:
        files = [ctx.attr.dummy]
    return [DefaultInfo(
        files = files,
        data_runfiles = ctx.runfiles(
            files = list(files),
        ),
    )]

"""
Provides all files (including `data` dependencies) at one level such that they
are expandable via `$(locations ...)`.

@param data
    Upstream data targets. This will consume both the `srcs` and `data`
    portions of an existing `filegroup`.
@param dummy
    Use this to avoid errors from empty "$(locations ...)" expansion.
    @ref https://github.com/bazelbuild/bazel/blob/c3bedec/src/main/java/com/google/devtools/build/lib/analysis/LocationExpander.java#L273  # noqa
"""

recursive_filegroup = rule(
    attrs = {
        "data": attr.label_list(
            cfg = "data",
            allow_files = True,
        ),
        "dummy": attr.label(allow_single_file = True),
    },
    implementation = _recursive_filegroup_impl,
)
