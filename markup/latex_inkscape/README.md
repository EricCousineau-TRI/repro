# Latex + Inkscape

I always forget a good workflow to do this. Capturing for myself, and to ask
for better workflows.

## Prereqs

Only tested on Ubuntu 18.04

```sh
sudo apt install \
    inkscape \
    texstudio \
    texlive-latex-recommended texlive-pictures texlive-latex-extra
```

References used:

* <https://tex.stackexchange.com/questions/158700/latex-cant-find-sty-files-altough-packages-are-installed-texlive-ubuntu-12/158721>

## Running

Just clone, go to this folder, then open:

```sh
texstudio ./main.tex
```

(ignore the equations, they're just there to fill space)

## Video

[mp4](https://user-images.githubusercontent.com/26719449/111078951-3f983080-84ce-11eb-823c-d9f16e0edfcf.mp4)

`mp4` recorded using [SimpleScreenRecorderer 0.3.8](https://github.com/MaartenBaert/ssr/releases/tag/0.3.8) (from Debian).

![gif](https://user-images.githubusercontent.com/26719449/111079241-85092d80-84cf-11eb-8bae-cd5dbdad055f.gif)

`gif` encoding using [gifski](https://gif.ski) 1.4.0:

```sh
gifski --fast-forward 2 --fast --fps 5 --width 1000 \
    2021-03-14-latex-inkscape-question-again.mp4 \
    --output 2021-03-14-latex-inkscape-question-again.gif
```

## Images

Before:

![before](./drawing-before.svg)

After:

![after](./drawing-after.svg)
