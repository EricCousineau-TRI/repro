#!/bin/bash
set -eu -o pipefail

# Runs glxgears for short period, then quits.
glxgears &
sleep 0.5
kill $(jobs -p)
echo "Success"
