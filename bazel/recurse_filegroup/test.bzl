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
      #  runtime. It is also possible to define data_runfiles and
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
