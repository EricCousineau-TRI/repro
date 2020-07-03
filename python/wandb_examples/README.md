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
# Show it working
bazel test --config=show_test :wandb_stub_test

# Show it failing
bazel test --config=show_test --test_arg="--show_error" :wandb_stub_test
```

Failure output:
```
Traceback (most recent call last):
  File "{runfiles}/wandb_examples/
test/wandb_stub_test.py", line 48, in test_wandb
    wandb.init(project="test_project", sync_tensorboard=True)
  File "{execroot}/wandb_examples/venv/lib/python3.6/site-packages/wandb/__init__.py", line 1129, in init
    _init_headless(run, False)
  File "{execroot}/wandb_examples/venv/lib/python3.6/site-packages/wandb/__init__.py", line 259, in _init_headless
    stdout_master_fd, stdout_slave_fd = io_wrap.wandb_pty(resize=False)
  File "{execroot}/wandb_examples/venv/lib/python3.6/site-packages/wandb/io_wrap.py", line 157, in wandb_pty
    master_fd, slave_fd = pty.openpty()
  File "/usr/lib/python3.6/pty.py", line 29, in openpty
    master_fd, slave_name = _open_terminal()
  File "/usr/lib/python3.6/pty.py", line 59, in _open_terminal
    raise OSError('out of pty devices')
OSError: out of pty devices
```

Bazel version:
```
$ bazel version
Build label: 3.0.0
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Mon Apr 6 12:52:37 2020 (1586177557)
Build timestamp: 1586177557
Build timestamp as int: 1586177557
```
