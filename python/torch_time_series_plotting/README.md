# Shows time series plotting with WandB+Tensorboard

## Setup

Tested manually on Ubuntu 18.04, assuming `apt install python3-venv`.

```sh
./setup.sh
```

## Running

```sh
# Run it.
./torch_time_series_plotting_example.py

# View it.
tensorboard --logdir=/tmp/time_series_example/log
```

## Example Results

* WandB Results: <https://wandb.ai/eacousineau/test-public/runs/6x6dfid1>
    * No images?

* TensorBoard.dev: <https://tensorboard.dev/experiment/9e4KIRueT0qcEswPpLQOHQ>
    * Also no images (yet)?

### Video

[mp4](https://user-images.githubusercontent.com/26719449/112756480-57ce7c00-8fb3-11eb-9915-f3d1a6c3e6a6.mp4)

![gif](https://user-images.githubusercontent.com/26719449/112756525-9e23db00-8fb3-11eb-92f7-749b43a57a59.gif)

#### Tools Used

`mp4` recorded using [SimpleScreenRecorderer 0.3.8](https://github.com/MaartenBaert/ssr/releases/tag/0.3.8) (from Debian).

Keyboard shown using `pip` installed version of specific commit for
[`key-mon`](https://github.com/scottkirkwood/key-mon/tree/3785370d0)

`gif` encoding using [gifski](https://gif.ski) 1.4.0:

```sh
gifski --fast-forward 2 --fast --fps 5 --width 1000 \
    <base>.mp4 --output <base>.gif
```
