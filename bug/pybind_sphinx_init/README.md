# pybind + sphinx init issue

For https://github.com/RobotLocomotion/drake/issues/11954

To repro, ensure you have Sphinx installed, and have the `pybind11` submodule checked out:

    cd repro/
    git submodule update --init -- externals/pybind11

Then in this directory run:

    cd repro/bug/pybind_sphinx_init/
    ./repro.sh

This will build two versions of the doc, one without the workaround, one with.
