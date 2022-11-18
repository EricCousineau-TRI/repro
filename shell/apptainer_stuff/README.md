# AppTainer

Shane Loretz told me about this, and I have been liking it a bit more than
Docker:

- No server
- Very easy to use with NVidia
- No crazy permission changes for mounted files, etc.
- Less mysterious `sudo` stuff (though still figuring that out)

However, I'm still getting a hold of it. This is me documenting my steps /
tools.

## Building `apptainer`

Building apptainer to run / administer without privilege: \
https://gist.github.com/EricCousineau-TRI/df9777773c6a6c07c06ddef2f4f82fa3

```sh
# See https://apptainer.org/docs/user/1.0/quick_start.html#quick-installation-steps

# From docs.
sudo apt install \
   build-essential \
   libseccomp-dev \
   pkg-config \
   squashfs-tools \
   cryptsetup \
   curl wget git
# For not needing `sudo` - uidmap needed for --fakeroot to work
sudo apt install uidmap

wget https://golang.org/dl/go1.18.linux-amd64.tar.gz -O /tmp/go.tar.gz
tar -xzf /tmp/go.tar.gz -C ~/.local/opt
ln -sf ~/.local/opt/go/bin/go ~/.local/bin

cd ~/devel/
git clone https://github.com/apptainer/apptainer -b v1.0.0
cd apptainer

./mconfig --without-suid -p ~/.local/opt/apptainer \
    && make -C builddir \
    && make -C builddir install -j 8
ln -s ~/.local/opt/apptainer/bin/apptainer ~/.local/bin/
```

## Jammy

```sh
$ apptainer build --fakeroot jammy.sif jammy.Apptainer
$ apptainer build --fakeroot --sandbox jammy.sif.sandbox jammy.sif
```

## Drake and ROS 2

Below based on \
https://gist.github.com/sloretz/074541edfe098c56ff42836118d94a8d

To install Drake, using binary for now: \
https://drake.mit.edu/from_binary.html#stable-releases

```sh
git clone https://github.com/sloretz/apptainer-ros

./build_image.sh jammy-ros-humble-desktop
./build_image.sh jammy-ros-humble-desktop-ext
./build_image.sh jammy-ros-rolling-desktop
./build_image.sh jammy-ros-rolling-desktop-ext
```

Should use `./bash_extra.sh` to launch containers.

```sh
apptainer-ros-jammy
# apptainer-ros-jammy --fakeroot
```

### drake-ros specifics

```sh
# Host
cd ~/Downloads
wget https://drake-packages.csail.mit.edu/drake/nightly/drake-20221116-jammy.tar.gz

# Containre.
rm -rf /opt/drake
mkdir /opt/drake
tar -xzf ~/Downloads/drake-20221116-jammy.tar.gz -C /opt/drake \
    --strip-components=1

/opt/drake/share/drake/setup/install_prereqs

cd .../ros_ws
mkdir -p src
ln -sf .../drake-ros ./src/
rm -rf build/ install/ log/

source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src -ryi

colcon build --packages-up-to drake_ros_core
colcon test --packages-up-to drake_ros_core
colcon test-result --verbose
```
