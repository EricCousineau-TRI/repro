## PyAssimp hacks

To build `pyassimp` from source:

```sh
cd assimp
# In assimp source tree.
git clone https://github.com/assimp/assimp -b v5.0.1
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
