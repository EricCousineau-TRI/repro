# VTK + `pybind11` Interop

Prototype code, triggered by:

* [This StackOverflow post](https://stackoverflow.com/questions/54871216/pybind11-return-c-class-with-an-existing-python-binding-to-python)
* Desired use in TRI project that uses [Drake](https://drake.mit.edu).

Generalized (and possibly more robust?) version of [SMTK VTK code](
https://gitlab.kitware.com/cmb/smtk/blob/9bf5b4f9/smtk/extension/vtk/pybind11/PybindVTKTypeCaster.h). This code was pointed out by `curtainsbaked`
in the StackOverflow post.

## Reproduction for Ubuntu 18.04

Following these steps will permit running unittests.

### Setup

    cd vtk_pybind
    install_dir=${PWD}/build_install

For each of the prereqs, clone somewhere, install prereqs.

### `pybind11`

    cd pybind11
    git checkout 25abf7e
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${install_dir} \
        -DPYBIND11_PYTHON_VERSION=3 \
        -DPYBIND11_TEST=OFF
    make install

### `vtk`

    cd vtk
    git checkout v8.2.0
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${install_dir} \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DVTK_WRAP_PYTHON=ON -DVTK_PYTHON_VERSION=3
    make install

### This project

    cd vtk_pybind
    mkdir build && cd build
    cmake .. -DCMAKE_PREFIX_PATH=${install_dir}
    make && ctest -V -R
