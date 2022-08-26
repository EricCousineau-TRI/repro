#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})
sleep_compare=../bazel-bin/timing_profiler/sleep_compare

(
    cat /proc/cpuinfo | grep 'model name' | uniq
    nproc

    ${sleep_compare} --timerslack_usec=0
    ${sleep_compare} --timerslack_usec=5
    chrt -r 20 ${sleep_compare} --timerslack_usec=0
    chrt -r 20 ${sleep_compare} --timerslack_usec=5
    taskset -c 0,6 chrt -r 20 ${sleep_compare} --timerslack_usec=0
    taskset -c 0,6 chrt -r 20 ${sleep_compare} --timerslack_usec=5
) 2>&1 | tee ./sleep_compare_example.output.txt
