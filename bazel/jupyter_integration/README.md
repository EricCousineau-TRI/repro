# Bazel-Jupyter Stuff

Example of using JupyterLab with Bazel's `rules_python`.

Known issues:

- Dunno how to make `jupyterlab-widgets` actually install correctly. See coarse
  notes in `/notes.md`.

See [tools/jupyter README](./tools/jupyter/README.md) for usage and examples.

## Setup

Ensure that you have Jupyter installed, and optionally Jupyter Lab.

If just using Jupyter on Ubuntu 18.04:

    sudo apt install jupyter-notebook

For Jupyter Lab, PIP is probably the easiest way.
