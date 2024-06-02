**NOTE**: This is an older / rougher approach.

See a *slightly* more general / more systems-like approach: \
<https://github.com/EricCousineau-TRI/drake_diff_ik_dual_arm>

Note that you should keep an eye out for a better formulation in Drake itself.

# Rough Drake Collision Avoidance Diff IK Version

This is a rough example, not very Systems-based. Stay tuned for better example.

## Setup

Be sure you have `poetry`: <https://python-poetry.org>
Then setup this folder.

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
python -m diff_ik_collision_avoidance.main
```
