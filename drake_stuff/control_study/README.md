# Control Study

Trying out various SE(3) controllers.

## Setup

Be sure you have `poetry`: <https://python-poetry.org>

```sh
poetry install
```

## Running

```sh
# For each terminal.
poetry shell

# (Separate terminal) Visualizer
python -m pydrake.visualization.meldis -w

# (Main terminal) Running
python -m control_study.main
```
