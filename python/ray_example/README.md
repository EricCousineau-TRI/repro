# Basic Test for ray

Tested on Ubuntu 20.04

Can configure for respective cloud, then test w/ the following command:

```sh
./run_all.sh
```

Meant to simulate provisioning using fully custom setup, not relying on `ray`'s
default `pip install --user` commands.

In this case, it shows using a custom `venv`, without leaking it into full
filesystem.
