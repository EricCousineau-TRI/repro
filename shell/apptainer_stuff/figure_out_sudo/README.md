# Figure out how to make `sudo` work in `apptainer` without host root permissions

```sh
# Skip using .sif, just go straight to persistent writeable sandbox.
# Generate writeable sandbox
$ apptainer build --fakeroot --sandbox image.sif.sandbox ./Apptainer

# Starting as user
$ apptainer --silent exec --writable image.sif.sandbox bash -c 'sudo echo Hello'
/usr/bin/bash: line 1: sudo: command not found

# Debugging
$ apptainer --silent exec --writable --fakeroot image.sif.sandbox bash --norc
$ apt install sudo
$ sudo echo Hey
sudo: unable to allocate pty: Operation not permitted

# Resume as user
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

