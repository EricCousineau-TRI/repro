#!/bin/bash
set -eu

echo "Host"
dpkg -s libc6 | grep Version

echo "Container"
apptainer --silent exec ./repro.sandbox bash -c 'dpkg -s libc6 | grep Version'

# Output:
#
# Host
# Version: 2.35-0ubuntu3.1
# Container
# Version: 2.31-0ubuntu9.9
