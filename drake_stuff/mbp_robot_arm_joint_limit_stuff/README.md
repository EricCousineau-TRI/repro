# Joint Limit Things

## Prereqs

Tested on Ubuntu 18.04 (Bionic). Needs ROS1 Melodic, Drake prereqs, and
pyassimp.

## Setup

You can just run this:

```sh
./setup.sh
```

This will:

* Set up a small `virtualenv` with Drake and JupyterLab
* Clone `ur_description` and, uh, convert it to format that Drake can use :(

## Running

```sh
./setup.sh jupyter lab ./joint_limits.ipynb
```

## PyAssimp hacks

```sh
cd assimp
src_dir=${PWD}
# install_dir=${src_dir}/build/install
install_dir=~/proj/tri/repo/repro/drake_stuff/mbp_robot_arm_joint_limit_stuff/venv
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${install_dir} -GNinja
ninja install

cd ${src_dir}/port/PyAssimp/
python3 ./setup.py install --prefix ${install_dir}

cd ${install_dir}/lib/python3.6/site-packages/pyassimp
ln -s ../../../libassimp.so ./
```
