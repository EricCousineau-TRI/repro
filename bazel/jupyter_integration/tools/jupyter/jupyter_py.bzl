load("//tools/skylark:generate_file.bzl", "generate_file")

# Generate file, because we wish to bake the file directly in, and not require
# it be passed as an argument.
_JUPYTER_PY_TEMPLATE = """
import os
import sys

from jupyter_bazel import _jupyter_bazel_notebook_main

from jupyter_integration.jupyter.runfiles import SubstituteMakeVariableLocation

notebook_path = SubstituteMakeVariableLocation({notebook_target})
_jupyter_bazel_notebook_main(notebook_path, sys.argv[1:])
""".lstrip()

def _jupyter_py_target(
        name,
        target = None,
        notebook = None,
        data = [],
        deps = [],
        tags = [],
        **kwargs):
    if target == None:
        fail("Must supply `target`")
    impl_file = "{}_run_notebook.py".format(name)
    notebook_target = "$(location {}//{}:{})".format(
        native.repository_name(),
        native.package_name(),
        notebook,
    )
    generate_file(
        name = impl_file,
        content = _JUPYTER_PY_TEMPLATE.format(
            notebook_target = repr(notebook_target),
        ),
        is_executable = False,
    )
    target(
        name = name,
        srcs = [impl_file],
        main = impl_file,
        data = data + [
            notebook,
        ],
        deps = deps + [
            "//tools/jupyter:jupyter_bazel_py",
        ],
        # `generate_file` output is still marked as executable :(
        tags = tags + ["nolint"],
        **kwargs
    )

def jupyter_py_binary(name, notebook = None, add_test_rule = 0, **kwargs):
    """Creates a target to run a Jupyter/IPython notebook.

    Please see `//tools/jupyter:README.md` for examples.

    @param notebook
        Notebook file to use. Be default, will be `{name}.ipynb`.
    """
    if notebook == None:
        notebook = name + ".ipynb"
    _jupyter_py_target(
        name = name,
        target = native.py_binary,
        notebook = notebook,
        **kwargs
    )
    if add_test_rule:
        jupyter_py_test(
            name + "_test",
            notebook = notebook,
            **kwargs
        )

def jupyter_py_test(name, notebook = None, tags = [], args = [], **kwargs):
    """Creates a target to test a Jupyter/IPython notebook.

    Please see `//tools/jupyter:README.md` for examples.

    @param notebook
        Notebook file to use. Be default, will be `test/{name}.ipynb`.
    """
    if notebook == None:
        notebook = "test/" + name + ".ipynb"
    _jupyter_py_target(
        name = name,
        tags = tags + ["jupyter"],
        target = native.py_test,
        args = ["--test"] + args,
        notebook = notebook,
        **kwargs
    )
