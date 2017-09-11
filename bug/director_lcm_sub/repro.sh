#!/bin/bash
set -x

{ ./sub.py; } &
job=$!

{
    # set +x
    ./pub.py
    ./pub.py --do_sleep
}

kill -s STOP $job
