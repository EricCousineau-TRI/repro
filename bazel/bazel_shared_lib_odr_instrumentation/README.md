# Relinking without recompilation

Motivation: Relinking C++ static binaries for shared libraries, for use with
Python.

e.g. workflow mentioned here:
https://github.com/bazelbuild/bazel/issues/16209

## Question 1

Can we compile cc files via one target, and expose `*.o` files via another
target externally?

Answer: yes. yay!

### Repro

```sh
$ bazel clean
$ bazel build :ex1_main_via_recompile
# Should see two warnings indicating two recompilations.
$ bazel clean
$ bazel build :ex1_main_via_object_files
# Should see only one warning.
```

## Question 2

Can we avoid any object files that may come in through dependencies? (e.g.
dreaded diamond dependency thing)

Answer: double yes. double yay!

```sh
$ bazel run :ex2_main -- libex2_b_bad.so.1 libex2_c_bad.so.1
wrapped_b: 1
wrapped_c: 1
FAIL! Separate memory for counter() -> ODR violation!

$ bazel run :ex2_main -- libex2_b_good.so.1 libex2_c_good.so.1
wrapped_b: 1
wrapped_c: 2
Success!
```
