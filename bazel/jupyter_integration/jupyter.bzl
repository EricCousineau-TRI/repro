
def _generate_file_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, ctx.attr.content, ctx.attr.is_executable)
    return [DefaultInfo(
        files = depset([out]),
        data_runfiles = ctx.runfiles(files = [out]),
    )]

# From Drake.
generate_file = rule(
    attrs = {
        "content": attr.string(mandatory = True),
        "is_executable": attr.bool(default = False),
    },
    output_to_genfiles = True,
    implementation = _generate_file_impl,
)

# Generate file, because we wish to bake the file directly in, and not require
# it be passed as an argument.
jupyter_template = """
#!/usr/bin/env python2

import os
import subprocess
import sys

# Assume that (hopefully) the notebook neighbors this generated file.
cur_dir = os.path.dirname(__file__)
notebook_file = "{notebook_file}"
notebook_path = os.path.join(cur_dir, notebook_file)
print(notebook_path)

# Determine if this is being run as a test, or via `./run`.
in_bazel = True
if "BAZEL_RUNFILES" in os.environ:
    # We are running via `./run`; we should be able to run without conversion.
    in_bazel = False
    print("Running direct notebook")
    os.chdir(os.environ["BAZEL_RUNFILES"])
    os.execvp("jupyter", ["jupyter", "notebook", notebook_path])
else:
    os.execvp("jupyter", ["jupyter", "nbconvert", "--execute", notebook_path])
"""

def _py_jupyter_target(name, target, notebook = None, data = [], **kwargs):
    impl_file = "{}_run_notebook.py".format(name)
    vars = dict(
        notebook_file = notebook,
    )
    generate_file(
        name = impl_file,
        content = jupyter_template.format(**vars),
    )
    target(
        name = name,
        srcs = [impl_file],
        main = impl_file,
        data = data + [
            notebook,
        ],
        **kwargs)

def py_jupyter_binary(name, notebook = None, **kwargs):
    if notebook == None:
        notebook = name + ".ipynb"
    _py_jupyter_target(
        name = name,
        target = native.py_binary,
        notebook = notebook,
        **kwargs)

def py_jupyter_test(name, notebook = None, **kwargs):
    if notebook == None:
        notebook = "test/" + name + ".ipynb"
    _py_jupyter_target(
        name = name,
        target = native.py_test,
        notebook = notebook,
        **kwargs)
