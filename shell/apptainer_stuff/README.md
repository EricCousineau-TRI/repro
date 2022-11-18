# Apptainer

Shane Loretz told me about this, and I have been liking it a bit more than
Docker:

- No server
- Very easy to use with NVidia
- No crazy permission changes for mounted files, etc.
- Less mysterious `sudo` stuff (though still figuring that out)

However, I'm still getting a hold of it. This is me documenting my steps /
tools.

## Build and Install Apptainer

Build Apptainer to not need root prvi

```sh
cd ~/tmp/apptainer_stuff
./build_and_install_apptainer.sh
```

Place `~/.local/bin` on your `PATH`. Concretely, open up `~/.bash_aliases` (**not** `~/.bashrc`), and add

```sh
export PATH=~/.local/bin:${PATH}
```

Please **do not** place this in `~/.bashrc`. Everything that puts junk here is
wrong, IMO.

If you are not confident in wrangling `PATH` / environment variable issues,
then be sure to comment out newly added stuff.

Also, if you have exiting `~/.local/bin`, please inspect to see if you have any
potential shadowing issues. If you do, fix it.

## Jammy

```sh
$ ./build_image.sh jammy
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

Should use `./bash_apptainer.sh` to launch containers using functions.

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
