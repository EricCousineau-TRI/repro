#!/bin/bash

export-prepend () 
{ 
    eval "export $1=\"$2:\$$1\""
}

env-extend () 
{ 
    local prefix=$1;
    export-prepend PYTHONPATH $prefix/lib;
    export-prepend PATH $prefix/bin;
    export-prepend LD_LIBRARY_PATH $prefix/lib;
    export-prepend PKG_CONFIG_PATH $prefix/lib/pkgconfig
}

VTK_ROOT=$TRI/proj/dart_impl/install/vtk
VTK_VERSION=vtk-5.10

export VTK_INCLUDE=$VTK_ROOT/include/$VTK_VERSION
export VTK_LIBDIR=$VTK_ROOT/lib/$VKT_VERSION

env-extend $VTK_ROOT
