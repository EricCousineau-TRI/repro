#!/bin/bash

# Provides ROS environment (and some Python dependencies) to execute the
# desired arbitrary commands.
source $(dirname $0)/devel/setup.bash

# Execute.
set -eux -o pipefail
exec "$@"
