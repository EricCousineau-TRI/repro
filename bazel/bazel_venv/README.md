# Example bazel venv

The following is an example workflow of using a proper `virtualenv` in Bazel.

This is a scrubbed / exported version used in TRI's Anzu codebase.

## Motivation

`rules_python` is great at providing granular deps, but
can break some assumption on how things are packages, e.g.:

- https://github.com/bazelbuild/rules_python/issues/2071 - `rerun` could not be correctly
  due to a bug in (re)implementing `virtualenv` behavior
- https://github.com/bazelbuild/rules_python/issues/408#issuecomment-1881404309 -
  running jupyter notebooks can be somewhat painful to setup the env / path / args correctly.

## See Also

- https://github.com/RobotLocomotion/drake/issues/8392#issuecomment-2264239096
- https://github.com/RobotLocomotion/drake/pull/21844

This is a slightly more modified example of the export from the second link.

## Example

```sh
./setup/install_prereqs.sh
bazel run //example:example_test
```
