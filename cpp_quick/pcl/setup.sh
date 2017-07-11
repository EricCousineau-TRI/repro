#!/bin/bash
set -e -x -u

url=https://raw.github.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd
name=$(basename $url)

curl $url -o build/$name
