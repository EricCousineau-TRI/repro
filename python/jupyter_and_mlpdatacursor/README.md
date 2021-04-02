# Playing with JupyterLab and `mpldatacursor`

## Setup

Tested manually on Ubuntu 18.04, assuming `apt install python3-venv`.

```sh
./setup.sh
```

## Running

```sh
# Run it.
./setup.sh juptyer lab ./example.ipynb
```

### Video

[mp4](https://user-images.githubusercontent.com/26719449/113374023-1e429b80-933a-11eb-8ba6-344eed540b9b.mp4)

![gif](https://user-images.githubusercontent.com/26719449/113374057-331f2f00-933a-11eb-9dfd-1f3d1d3d1269.gif)

#### Tools Used

`mp4` recorded using [SimpleScreenRecorderer 0.3.8](https://github.com/MaartenBaert/ssr/releases/tag/0.3.8) (from Debian).

Keyboard shown using `pip` installed version of specific commit for
[`key-mon`](https://github.com/scottkirkwood/key-mon/tree/3785370d0)

`gif` encoding using [gifski](https://gif.ski) 1.4.0:

```sh
gifski --fast-forward 2 --fast --fps 5 --width 1000 \
    <base>.mp4 --output <base>.gif
```
