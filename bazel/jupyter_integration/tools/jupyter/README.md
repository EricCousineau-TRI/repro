# Simple Bazel-Jupyter Integration

This shows simple Bazel-Jupyter integration. Goals:

*   Permit running interactive notebook sessions using Bazel Python
dependencies, editing the original notebook outside of Bazel's sandbox.
*   Permit running the notebook via `bazel run`.
*   Permit running the notebook via `bazel test`.

## Browsing Notebooks

You can browse and edit all notebooks with the version of Jupyter that's
bundled in this workspace:

    bazel build //tools/jupyter:jupyter
    ./bazel-bin/tools/jupyter/jupyter notebook

Note that you may not be able to run anything, as it may not include all of
the target-specific dependencies.

## Creating a Notebook

If you wish to create a notebook, simply copy `./tools/jupyter/template.ipynb`
to the package of your choice, keeping the first cell. Next, you will need to
add the notebook to your package's `BUILD.bazel` file; see below for how to
do this.

## Adding Bazel Targets

The file `jupyter_py.bzl` contains the Skylark macro `jupyter_py_binary`, which
defines a `native.py_binary` target underneath. (`jupyter_py_test` also exists,
but is recommended to just use `jupyter_py_binary(..., add_test_rule = 1)`.)
Ensure that you add the appropriate Python dependencies (`deps = [...]`) and
data dependencies (`data = [...]`).

Some examples:

    load("//tools/jupyter:jupyter_py.bzl", "jupyter_py_binary")

    jupyter_py_binary(
        name = "example",
        deps = ...,
        add_test_rule = 1,
    )

## Running Notebooks with Bazel Dependencies

To run a notebook interactively, being able to save the notebook and still
access Bazel dependencies, use `./bazel-bin` or `bazel run`.
As an example in `bash`:

    # N.B. You must manually re-build if you've changed the Bazel targets.
    bazel run //tools/workspace:example

If you save the notebook, it will save to the original file (since it is a
symlink).

To run a test notebook non-interactively for testing, use `bazel test`:

    bazel test //tools/jupyter:example_test

Note that this will generally not output anything to the screen aside from
errors. This should be used as a way to see if your notebook has been broken.

## Committing to Git

If you commit a Jupyter notebook, please ensure that you clear all output
cells.

The easiest way to do this is make a keyboard shortcut:

*   Within a notebook, select `Help > Edit Keyboard Shortcuts`.
*   Scroll to `clear all cells output`, click `+`, and type `K,K` (or whatever
shortcut you want), press Enter, and click OK.
*   When you are saving your notebook to commit, to clear output, you can press
`ESC + K + K` to clear the outputs. (This does not change the state of your
kernel, so you can still access variables that you're working with.)

[//]: # "TODO(eric.cousineau): If there is a safe way to tie a `git commit` "
[//]: # "to a bazel binary, we should use it. Otherwise, just rely on review."
