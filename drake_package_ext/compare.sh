#!/bin/bash
set -e -u

{
    ./test.sh
    ./test.sh _install_compare
} | sed "s#$USER#user#g"
