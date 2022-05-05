# Joint Limit Things

## Prereqs

Tested on Ubuntu 20.04 (Focal).

- Install Ignition Gazebo (Model Photo Shoot plugin is needed):
  <https://ignitionrobotics.org/docs/latest/install>
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

# You can run the setup, transformation and tests through:
cd ..
./setup_transform_test.sh ${PWD}/repos/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2/ model.sdf

# Or you can run each step separately:
cd ..
source setup.sh

./format_model_and_generate_manifest.sh ${PWD}/repos/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2/ model.sdf

./compare_model_via_drake_and_ingition_images.sh ${PWD}/repos/CERBERUS_ANYMAL_C_SENSOR_CONFIG_2/ model.sdf

```

