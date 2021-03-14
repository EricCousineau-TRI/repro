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

![video](./video.mp4)

## Images

Before:

![before](./drawing-before.svg)

After:

![after](./drawing-after.svg)
