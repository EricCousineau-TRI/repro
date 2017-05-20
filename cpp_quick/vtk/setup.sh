#!/bin/bash

export-prepend () 
{ 
    eval "export $1=\"$2:\$$1\""
}

VTK_ROOT=$TRI/proj/dart_impl/install/vtk
VTK_VERSION=vtk-5.10

export VTK_INCLUDE=$VTK_ROOT/include/$VTK_VERSION
export VTK_LIBDIR=$VTK_ROOT/lib/$VTK_VERSION

export-prepend LD_LIBRARY_PATH $VTK_LIBDIR;
