#!/bin/bash

cd $(dirname $0)

docker run --rm -it -v ${PWD}:/working ubuntu:18.04 bash -c 'cd /working && ./repro.sh'
