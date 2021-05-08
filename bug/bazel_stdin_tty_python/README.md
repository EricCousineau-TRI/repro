# `bazel run`, tests, and stdin via tty

stdin redirect (`0<&0`, I think?) prolly messes it up?

## Run

```sh
$ bazel run :read_from_tty
Press enter... <Enter>
Done!

$ bazel run :read_from_tty_as_test
# May not print anything; gotta Ctrl+C out :(
```

## Info

```sh
$ bazel version
Build label: 4.0.0
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Thu Jan 21 07:33:24 2021 (1611214404)
Build timestamp: 1611214404
Build timestamp as int: 1611214404
```
