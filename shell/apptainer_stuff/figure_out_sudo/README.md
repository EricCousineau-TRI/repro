# Figure out how to make `sudo` work in `apptainer` without host root permissions

## Old Setup

When trying to install `sudo` package

```sh
# Skip using .sif, just go straight to persistent writeable sandbox.
# Generate writeable sandbox
$ apptainer build --force --fakeroot --sandbox image.sif.sandbox ./Apptainer

# As root
$ apptainer --silent exec --writable --fakeroot image.sif.sandbox \
    env -i bash -c 'sudo echo Hey'
sudo: unable to allocate pty: Operation not permitted

# As user
$ apptainer --silent exec --writable image.sif.sandbox \
    env -i bash -c 'sudo echo Hello'
sudo: /etc/sudo.conf is owned by uid 1002, should be 0
sudo: The "no new privileges" flag is set, which prevents sudo from running as root.
sudo: If sudo is running in a container, you may need to adjust the container configuration to disable the flag.
```

Questions:

* Should I just stub out `sudo`, e.g. make it do `exec "$@"`?
  That won't change permissions.
* Perhaps I always use `--fakeroot`, but then somehow fake out `$USER` /
  `whomai`?

## Stubbing `sudo` with `fake_sudo.sh`

Seems OK if always running via `--fakeroot`:

```sh
$ apptainer build --force --fakeroot --sandbox image.sif.sandbox ./Apptainer
$ apptainer --silent exec --writable --fakeroot image.sif.sandbox \
    env -i bash -c 'sudo echo Hey'
```

Good enough for testing user workflows with installs :shrug:
