#!/bin/bash
set -e -u

cd $(dirname $0)
outdir=$1
cd $outdir
wget https://www.opengl.org/archives/resources/code/samples/glut_examples/examples/blender.c
