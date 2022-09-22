https://gist.github.com/EricCousineau-TRI/df9777773c6a6c07c06ddef2f4f82fa3

but to install Drake, just do binary
https://drake.mit.edu/from_binary.html#stable-releases

```sh
git clone https://github.com/sloretz/apptainer-ros

# build *.sif
base=jammy-ros-humble-desktop
apptainer build --fakeroot ${base}.sif \
    ./apptainer-ros/jammy-ros-humble-desktop/Apptainer
# generate writeable / persistent sandbox
apptainre build --sandbox ${base}.sif.sandbox ${base}.sif

# execute with minimal containment
# - as root
apptainer --silent exec \
    --fakeroot --nv --writable \
    --pwd ${PWD} \
    jammy-ros-humble-desktop.sif.sandbox bash
# - as user
apptainer --silent exec \
    --nv --writable \
    --pwd ${PWD} \
    jammy-ros-humble-desktop.sif.sandbox bash
```

### workarounds

as root

```sh
ln -s /root /home/<user>
alias sudo=""  # pty failure?!
```

as user
```sh

# sudo fails tho
```

### drake-ros specifics

https://gist.github.com/sloretz/074541edfe098c56ff42836118d94a8d

```sh
# as root
mkdir /opt/drake
tar -xvzf ~/Downloads/drake-20220822-jammy.tar.gz -C /opt/drake --strip-components=1

/opt/drake/share/drake/setup/install_prereqs

cd .../ros_ws
mkdir src
ln -s .../drake-ros ./src/

source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src -ryi

# as user
source /opt/ros/humble/setup.bash
colcon build --packages-up-to drake_ros_examples
```
