A hacky(?) way to write sort-of-unittests for Starlark.

To see it pass, go to this directory, then simply
```
bazel build ...
```

To see what failure looks like, just set `c_expected` to a bogus value.
