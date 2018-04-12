#!/bin/bash

rm -rf ./build
mkdir build

virtualenv --system-site-packages build
source build/bin/activate
python setup.py -v install

python cos_module_np_example.py
