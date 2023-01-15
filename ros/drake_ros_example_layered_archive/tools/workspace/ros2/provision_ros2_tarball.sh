#!/bin/bash
set -eux -o pipefail

cd $(dirname ${BASH_SOURCE})

# Delete existing downloads.
rm -f ./*.tar.bz2* ./*CHECKSUM*

# Focal
# Download checksum via HTTPS.
wget https://repo.ros2.org/ci_archives/drake-ros-underlay/ros2-humble-linux-focal-amd64-ci-CHECKSUM
# Download the latest archive.
wget http://repo.ros2.org/ci_archives/drake-ros-underlay/ros2-humble-linux-focal-amd64-ci.tar.bz2
# Ensure checksum is correct.
sha256sum -c ./ros2-humble-linux-focal-amd64-ci-CHECKSUM
# Remove checksum.
rm ./ros2-humble-linux-focal-amd64-ci-CHECKSUM

# Jammy
# Download checksum via HTTPS.
wget https://build.ros2.org/view/Hci/job/Hci__nightly-cyclonedds_ubuntu_jammy_amd64/lastSuccessfulBuild/artifact/ros2-humble-linux-jammy-amd64-ci-CHECKSUM
# Download the latest archive.
wget https://build.ros2.org/view/Hci/job/Hci__nightly-cyclonedds_ubuntu_jammy_amd64/lastSuccessfulBuild/artifact/ros2-humble-linux-jammy-amd64-ci.tar.bz2
# Ensure checksum is correct.
sha256sum -c ./ros2-humble-linux-jammy-amd64-ci-CHECKSUM
# Remove checksum.
rm ./ros2-humble-linux-jammy-amd64-ci-CHECKSUM

# Upload archives.
cd ../../..
./tools/external_data/cli upload \
    ./tools/workspace/ros2/ros2-humble-linux-focal-amd64-ci.tar.bz2
./tools/external_data/cli upload \
    ./tools/workspace/ros2/ros2-humble-linux-jammy-amd64-ci.tar.bz2
