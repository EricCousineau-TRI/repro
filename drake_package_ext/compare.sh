#!/bin/bash
set -e -u

# Prequisite: Set $DRAKE environment variable
{
    # Force git to report a dirty status
    touch $DRAKE/dirty_file

    ./test.sh
    ./test.sh _install_compare
    ./test.sh _rsync
} | sed "s#$USER#user#g"
