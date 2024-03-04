# Bazel Python Ordering

Show that import order aligns with ordering in `deps`.

Can cause an issue based on transitive deps, etc.

```sh
$ bazel build //...
$ bazel-bin/import_order_1
my_lib.my_func() = 'from pkg_1'
$ bazel-bin/import_order_2
my_lib.my_func() = 'from pkg_2'
```
