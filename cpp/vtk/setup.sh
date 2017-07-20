#!/bin/bash

VTK_VERSION=vtk-5.10
# Squirrel this away wherever you like.
VTK_ROOT=~/.local/vtk/$VTK_VERSION
# Used by vtk_repository
export VTK_INCLUDE=$VTK_ROOT/include/$VTK_VERSION
export VTK_LIBDIR=$VTK_ROOT/lib/$VTK_VERSION
# Necessary for execution
export LD_LIBRARY_PATH=$VTK_LIBDIR:$LD_LIBRARY_PATH
