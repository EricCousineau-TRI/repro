# Simple Bazel-Jupyter Integration

This shows simple Bazel-Jupyter integration. Goals:

*   Permit running interactive notebook sessions using Bazel Python
dependencies, editing the original notebook outside of Bazel's sandbox.
*   Permit running the notebook via `bazel run`.
*   Permit running the notebook via `bazel test`.

## Usage:

The file `jupyter_py.bzl` contains the Skylark macros `jupyter_py_binary` and
`jupyter_py_test`, which defines `native.py_{binary,test}` targets underneath.

Pulling examples from `BUILD`:

    load(
        ":jupyter_py.bzl",
        "py_jupyter_binary",
        "py_jupyter_test",
    )

    py_library(
        name = "my_library",
        srcs = ["src/my_library.py"],
        imports = ["src"],
    )

    py_jupyter_binary(
        name = "simple_notebook",
        deps = [":my_library"],
    )

    py_jupyter_test(
        name = "simple_notebook_test",
        deps = [":my_library"],
    )

To run a notebook interactively, being able to save the notebook and still
access Bazel dependencies, use the `./run` script. As an example, in `bash`:

    # N.B. You must manually re-build if you've changed the Bazel targets.
    bazel build ...
    ./run //:simple_notebook

If you save the notebook, it will save to the original file.

*TODO(eric.cousineau)*: Figure out if `bazel run --direct` can enable this workflow.

To run a notebook as a `bazel run` or `bazel test`, just use those commands, e.g.:

    bazel run //:simple_notebook
    bazel test //:simple_notebook_test
