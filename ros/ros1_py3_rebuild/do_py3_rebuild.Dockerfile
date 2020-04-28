# Provides container that `do_py3_rebuild.py` can run in.
FROM ros:melodic-ros-core-bionic

# Install the Python 2 flavor of the packages we are going to rebuild,
# so that their dependencies also get installed.  We won't end up
# using the python2-specific dependencies, but at least this covers
# the version-agnostic dependencies.
RUN apt-get update && apt-get install -q -y \
    ros-melodic-cv-bridge \
    ros-melodic-tf2-py \
    && rm -rf /var/lib/apt/lists/*

# Install the Python 3 flavor of the dependencies of the modules
# listed in module_list.py.
RUN apt-get update && apt-get install -q -y \
    python3-numpy \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# N.B. `python3-catkin-pkg` seems to conflict with Python 2 version in `apt`.
# Use PIP instead.
RUN python3 -m pip install --no-cache-dir \
    vcstool==0.1.40 catkin_pkg==0.4.12
