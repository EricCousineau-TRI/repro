# Latex + Inkscape

I always forget a good workflow to do this. Capturing for myself, and to ask
for better workflows.

Update: Should've read the docs:

* <https://wiki.inkscape.org/wiki/index.php/LaTeX#How_to_embed_a_LaTeX_equation_inside_Inkscape>

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

Using Inkscape:

```sh
$ inkscape --version
Inkscape 0.92.3 (2405546, 2018-03-11)
```

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

## Bug: Paste between windows

Filed issue: <https://gitlab.com/inkscape/inbox/-/issues/4622>

[bug mp4](https://user-images.githubusercontent.com/26719449/111192837-e55fa400-858f-11eb-93a7-bb4f284a04b3.mp4)

![bug image](https://user-images.githubusercontent.com/26719449/111192999-150eac00-8590-11eb-9f7d-9b300844cba2.png)
