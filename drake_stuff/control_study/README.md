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

## Using `pydrake` with Proprietary Solvers

This should only be necessary for GUROBI. Build `drake` locally (do not redistribute!)
with Gurobi enabled, and install somewhere.

Here's a super hacky example, based on Python 3.10, Drake v1.15.0

- <https://drake.mit.edu/from_source.html>
- <https://drake.mit.edu/bazel.html#proprietary-solvers>

```sh
# In vanilla environment.
cd drake
git checkout v1.15.0
# WARNING: Bazel installation may not be stable. Use CMake to be safe.
bazel run --config=snopt --config=mosek --config=gurobi //:install -- ~/.local/opt/drake/v1.15.0

# In poetry.
poetry shell
pydrake_dir=$(python -c 'import os; import pydrake as m; print(os.path.dirname(m.__file__))')
mv ${pydrake_dir} ${pydrake_dir}_old
ln -s ~/.local/opt/drake/v1.15.0/lib/python3.10/site-packages/pydrake ${pydrake_dir}
```
