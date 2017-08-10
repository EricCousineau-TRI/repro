# Setup

To install everything together on Ubuntu 16.04:

    {
        sudo apt install git

        # Clone source code
        # drake
        git clone https://github.com/RobotLocomotion/drake
        # experimental
        git clone https://github.com/EricCousineau-TRI/repro
        (
            cd repro
            git submodule update --init -- externals/pybind11
        )

        # Install prequisites for both
        sudo ./repro/setup/ubuntu/16.04/install_prereqs.sh
        # - Defer to whatever drake installs anew
        sudo ./drake/setup/ubuntu/16.04/install_prereqs.sh

        # Build the baseline stuff in Drake
        cd drake
        bazel test //drake/bindings/pydrake_mathematical_program_test
    }

You will need to ensure you have Gurobi appropriately configured (and possibly SNOPT and MOSEK).

If the above test fails, you should fix it before moving on.

# Running

To run the example, use the following script to open MATLAB up with the correct environment (replace `${base}` with the correct path):

    cd ${base}/drake
    export DRAKE=${PWD}
    cd ${base}/repro
    ./python/bindings/mlmodule/demo/matlab_with_pythonpath_drake.sh

Then try out the two MATLAB scripts:

* `example_drake` - This does not wrap Drake's Python bindings.
* `example_drake_proxy` - This wraps Drake's Python bindings to make things easier.
    * This might be the easiest one to work with.
