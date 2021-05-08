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

### Video

[mp4](https://user-images.githubusercontent.com/26719449/117541256-01585280-afe1-11eb-8d51-77cf04e2e8dd.mp4)

![gif](https://user-images.githubusercontent.com/26719449/117541252-fe5d6200-afe0-11eb-8d86-63e569abf5e5.gif)

#### Tools Used

`mp4` recorded using [SimpleScreenRecorderer 0.3.8](https://github.com/MaartenBaert/ssr/releases/tag/0.3.8) (from Debian).

Keyboard shown using `pip` installed version of specific commit for
[`key-mon`](https://github.com/scottkirkwood/key-mon/tree/3785370d0)

`gif` encoding using [gifski](https://gif.ski) 1.4.0:

```sh
gifski --fast-forward 2 --fast --fps 5 \
    <base>.mp4 --output <base>.gif
```
