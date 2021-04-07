# Goal: Jupyter, Interactive Updates, Static Matplotlib, and Google Colab

See how they differ and all that.

## Running Locally

Tested manually on Ubuntu 18.04, assuming `apt install python3-venv`.

Using venv:

```sh
./setup.sh jupyter lab ./example.ipynb
```

With drake apt prereqs:

```sh
jupyter notebook ./example.ipynb
```

#### Recording Tools

`mp4` recorded using [SimpleScreenRecorderer 0.3.8](https://github.com/MaartenBaert/ssr/releases/tag/0.3.8) (from Debian).

Keyboard shown using `pip` installed version of specific commit for
[`key-mon`](https://github.com/scottkirkwood/key-mon/tree/3785370d0)

`gif` encoding using [gifski](https://gif.ski) 1.4.0:

```sh
gifski --fast-forward 2 --fast --fps 5 --width 1000 \
    <base>.mp4 --output <base>.gif
```
