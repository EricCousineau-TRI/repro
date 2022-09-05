# Anzu's Style of Defining `pybind11` Bindings w/ Drake

These is a simple, but hopefully relevant, export from TRI's Anzu codebase (as
of 2022-09-05), with a small amount of scrubbing applied.

## Context

* [Drake Developer Docs: Python Bindings](https://github.com/RobotLocomotion/drake/blob/v1.7.0/bindings/pydrake/pydrake_doxygen.h#L166-L186)
* [Drake External Examples: `drake_bazel_external/apps/BUILD.bazel`](https://github.com/RobotLocomotion/drake-external-examples/blob/d7a9a4331d2eb5fcb0dafdd47cc9b92834423cc4/drake_bazel_external/apps/BUILD.bazel#L82-L101)

For definitions of "ODR robust", see docs for `anzu_cc_shared_library()` in
`anzu_pybind.bzl`.

## Example

For working example:

```sh
bazel run //common:cc_py_test
```

You can see error messages by reviewing "Testing" section in
`anzu_pybind.bzl`.

## Linting

**NOTE**: Linting is not instrumented to work.

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
