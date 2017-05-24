# Note

Sources adapted from:

* https://github.com/RobotLocomotion/drake/tree/e132af9/drake/bindings

# Structure

* `bindings/`
    * `README.md` - instructions.
    * `pydrake/`
        * `_pydrake_typebinding.cc` - `pybind11` module file.
        * `_util/py_relax.h` - relaxed casting prototype.
        * `test/testTypeBinding.py` - unittest showing behavior.
    * `mldrake/`
        * `PyProxy.m` - hacky proxy wrapper
        * `test/`
            * `matlab_with_pythonpath.sh` - will build and source environment to use Bazel's Python modules, and start MATLAB with the path setup.
            * `TypeBindingTest.m` - MATLAB unittest.
            * `run_all.m` - Run example script and unittest.
    * `install_prereqs.sh` - minimal install for using Bazel, tested on basic Docker image (see `:/shell/docker/docker-x11.sh`).

# pybind11 within MATLAB using Bazel

To see Python bindings used from within MATLAB, in `bash`:

    $ ./mldrake/test/matlab_with_pythonpath.sh

and in MATLAB:
    
    >> run_all
