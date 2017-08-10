# Setup

To install everything together on Ubuntu 16.04:

    {
        sudo apt install git

        # Clone information
        git clone https://github.com/RobotLocomotion/drake
        git clone https://github.com/EricCousineau-TRI/repro
        (
            cd repro
            git submodule update --init -- externals/pybind11
        )

        # Install prequisites for both
        sudo ./repro/setup/ubuntu/16.04/install_prereqs.sh
        # - Defer to whatever drake installs anew
        sudo ./drake/setup/ubuntu/16.04/install_prereqs.sh

        # Build and run the demo, with MATLAB
        ./repro/python/bindings/mlmodule/demo/matlab_with_pythonpath_drake.sh
    }
