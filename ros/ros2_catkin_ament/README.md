# Catkin + Ament under One Roof

```
source /opt/ros/crystal/setup.bash
source /opt/ros/melodic/setup.bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/opt/ros/crystal;/opt/ros/melodic"
make
```
