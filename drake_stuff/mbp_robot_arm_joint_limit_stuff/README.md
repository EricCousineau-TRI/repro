# Joint Limit Things

## Prereqs

Tested on Ubuntu 20.04 (Focal).

- Install Gazebo Classic:
  <http://gazebosim.org/tutorials?tut=install_ubuntu&cat=install>
- Ensure Drake prereqs are installed:
  <https://drake.mit.edu/from_binary.html#stable-releases>
- Install additional packages:
  `sudo apt install imagemagick`

## Example

For rendering CERBERUS:
<https://app.ignitionrobotics.org/OpenRobotics/fuel/models/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2/6>

Download archive to `/tmp/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2.zip`.

```sh
# Download data.
mkdir -p repos && cd repos
# Manually download archive to /tmp/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2.zip
unzip /tmp/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2.zip -d ./CERBERUS_ANYMAL_C_SENSOR_CONFIG_2/

# Run setup.
cd ..
./setup.sh ${PWD}/repos/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2/ model.sdf
```
