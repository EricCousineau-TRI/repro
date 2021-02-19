# `wandb` examples / repro

## Ubuntu Deps

* Bazel 3.0.0
* `python3-virtualenv`

## config nesting for sweeps

<https://github.com/wandb/client/issues/982#issuecomment-652766322>

Files:

* `containers.py`
* `containers_test.py`

To view:

```
bazel run :containers_test
```

## `wandb` and `pkg_resources`

<https://github.com/wandb/client/issues/1101>

Unable to repro here :(

## `wandb` and `bazel test`

<https://github.com/wandb/client/issues/1137>

Files:

* `requirements.txt`
* `test/wandb_stub_test.py`

To reproduce:

```sh
# No longer fails!
$ bazel test --config=show_test :wandb_stub_test


```

Bazel version:
```
$ bazel version
Build label: 3.7.1
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Tue Nov 24 17:38:30 2020 (1606239510)
Build timestamp: 1606239510
Build timestamp as int: 1606239510
```
