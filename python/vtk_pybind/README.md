# Reproduction for Ubuntu 18.04

    cd vtk_pybind
    install_dir=${PWD}/build_install

pybind:

    cd pybind11
    git checkout 25abf7e
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${install_dir} \
        -DPYBIND11_PYTHON_VERSION=3 \
        -DPYBIND11_TEST=OFF
    make install

vtk:

    cd vtk
    git checkout v8.2.0
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=${install_dir} \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DVTK_WRAP_PYTHON=ON -DVTK_PYTHON_VERSION=3
    make install

This project:

    cd vtk_pybind
    mkdir build && cd build
    cmake .. -DCMAKE_PREFIX_PATH=${install_dir}
    make
    ctest -V -R
