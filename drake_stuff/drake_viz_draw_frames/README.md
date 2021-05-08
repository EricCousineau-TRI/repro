# Drawing Frames to Drake Visualizer in Python

All commands assumed to be run from this directory.

Only tested on Ubuntu 18.04 (Bionic).

## Setup

```sh
./setup.sh
```

You may need to install prereqs after installed:

```sh
sudo ./venv/share/drake/setup/install_prereqs
```

## Run Test

In each terminal, assumes you either prefix each command with `./setup.sh` or you
call `source ./setup.sh`.

```sh
# Terminal 1
drake-visualizer

# Terminal 2
python3 ./test/director_client_frames_test.py
```
