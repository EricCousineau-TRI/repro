#!/bin/bash

to_patch() {
    git format-patch -1 ${1} --stdout
}

cur=$(cd $(dirname $BASH_SOURCE) && pwd)
cd /home/eacousineau/devel/util/numpy
to_patch 7e0ca4b > ${cur}/feature_v1.11.0.patch
to_patch 71de985 > ${cur}/feature_v1.15.0.dev0+40ef8a6.patch
to_patch 6039494 > ${cur}/feature_indicator.patch
