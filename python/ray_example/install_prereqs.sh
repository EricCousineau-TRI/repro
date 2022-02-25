#!/bin/bash
set -eux

# Remove annoying stuff.
sudo apt-get -y purge unattended-upgrades

# Provision virtualenv tooling.
sudo apt-get update
sudo apt-get -y install python3-venv
