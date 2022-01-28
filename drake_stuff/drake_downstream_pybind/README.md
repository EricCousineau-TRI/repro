# Anzu's Style of Defining `pybind11` Bindings w/ Drake

These is a simple, but hopefully relevant, export from TRI's Anzu codebase (as
of 2022-01-28), with a small amount of scrubbing applied.

## Context

* [Drake Developer Docs: Python Bindings](https://github.com/RobotLocomotion/drake/blob/v0.38.0/bindings/pydrake/pydrake_doxygen.h#L166-L186)
* [Drake External Examples: `drake_bazel_external/apps/BUILD.bazel`](https://github.com/RobotLocomotion/drake-external-examples/blob/85ab0c925cc9a3789998e0ff0db2417319c66065/drake_bazel_external/apps/BUILD.bazel#L82-L101)

For definitions of "ODR robust", see docs for `anzu_cc_shared_library()` in
`anzu_pybind.bzl`.

## Files

* `anzu_pybind.bzl`; we place this in `anzu/tools/starlark/`
* `anzu_pybind_bazel_check.py`; we place this in `anzu/tools/lint/`

## Linting

To integrate linting, we have our nominal `add_lint_tests()` (defined in
`tools/lint/ilnt.bzl`, where we add our lint check as so:

```py
...  # other lint imports
load("//tools/skylark:anzu_pybind.bzl", "anzu_pybind_bazel_lint")

def add_lint_tests(
    ...  # other lint args
    anzu_pybind_bazel_lint_exclude = [],
    anzu_pybind_bazel_lint_extra_srcs = [],
):
    ...  # other lint rules
    anzu_pybind_bazel_lint(
        exclude = anzu_pybind_bazel_lint_exclude,
        extra_srcs = anzu_pybind_bazel_lint_extra_srcs,
    )
    ...
```
