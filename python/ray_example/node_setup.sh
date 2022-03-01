#!/bin/bash
set -eux

sudo apt-get -y purge unattended-upgrades
sudo apt-get update
sudo apt-get -y install python3-venv

cd ~/ray_stuff
./log_persist.sh
./setup.sh

# Make env be automatic.
echo 'source ~/ray_stuff/setup.sh' > ~/.bash_profile
