#!/bin/bash
set -e -u

{
    ./test.sh
    ./test.sh _install_compare
    ./test.sh _rsync
} | sed "s#$USER#user#g"
