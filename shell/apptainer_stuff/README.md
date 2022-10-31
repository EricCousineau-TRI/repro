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

## Drake and ROS 2

Below based on \
https://gist.github.com/sloretz/074541edfe098c56ff42836118d94a8d

To install Drake, using binary for now: \
https://drake.mit.edu/from_binary.html#stable-releases

```sh
git clone https://github.com/sloretz/apptainer-ros

# Build *.sif
image=jammy-ros-humble-desktop
apptainer build --fakeroot ${image}.sif \
    ./apptainer-ros/jammy-ros-humble-desktop/Apptainer
# Generate writeable sandbox with modifications.
image_ext=jammy-ros-humble-desktop-ext
apptainer build --fakeroot --sandbox ${image_ext}.sif.sandbox ${image_ext}.Apptainer

# Execute with minimal containment
# - as root, for install prereqs.
apptainer --silent exec \
    --nv --writable \
    --pwd ${PWD} \
    --fakeroot \
    ${image}.sif.sandbox \
    bash
# - as user, for nominal usage.
apptainer --silent exec \
    --nv --writable \
    --pwd ${PWD} \
    ${image}.sif.sandbox \
    bash
# TODO(eric.cousineau): See below for wanting to simplify usage of `sudo`.
```

or using `./bash_extra.sh` in whatevs dir

```sh
apptainer-ros-jammy
# apptainer-ros-jammy --fakeroot
```

### drake-ros specifics

```sh
# As root
mkdir /opt/drake
tar -xvzf ~/Downloads/drake-20220822-jammy.tar.gz -C /opt/drake --strip-components=1

/opt/drake/share/drake/setup/install_prereqs

cd .../ros_ws
mkdir src
ln -s .../drake-ros ./src/

source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src -ryi

# As user
source /opt/ros/humble/setup.bash
colcon build --packages-up-to drake_ros_examples
colcon test --packages-up-to drake_ros_examples
colcon test-results --verbose
```
