#!/bin/bash
set -eu -o pipefail

count=0
while :; do
    echo ${count}
    sleep 0.05
    let count=count+1
done
