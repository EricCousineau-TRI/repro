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
