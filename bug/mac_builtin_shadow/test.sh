#!/bin/bash
set -eux

# Purpose: Reproduce https://github.com/RobotLocomotion/drake/issues/8041

cd $(dirname $0)

rm -rf ./build
mkdir build
cd build

mkdir example_module
cd example_module
touch __init__.py

use_py=${USE_PY:-}

if [[ -z ${use_py} ]]; then

echo "Using shared library"

cat > math.c <<EOF
// https://docs.python.org/2/extending/extending.html

#include <Python.h>

static PyObject *
spam_system(PyObject *self, PyObject *args)
{
const char *command;
int sts;

if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;
sts = system(command);
return Py_BuildValue("i", sts);
}

static PyMethodDef SpamMethods[] = {
{"stuff",  spam_system, METH_VARARGS,
 "Execute a shell command."},
{NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initmath(void)
{
(void) Py_InitModule("math", SpamMethods);
}
EOF

cat > _math_build.py <<EOF
# https://docs.python.org/2/extending/building.html#building

from distutils.core import setup, Extension

module1 = Extension('math',
                sources = ['math.c'])

setup (name = 'PackageName',
   version = '1.0',
   description = 'This is a demo package',
   ext_modules = [module1])
EOF

python _math_build.py build
cp $(find . -name '*.so') .

else

echo "Using .py"

cat > math.py <<EOF
stuff = 1
EOF

fi


cat > other.py <<EOF
from __future__ import absolute_import

from math import log
from .math import stuff
EOF

cat > import_test.py <<EOF
from example_module.other import log, stuff
print(log, stuff)
EOF

cd ..
export PYTHONPATH=${PWD}:${PYTHONPATH}

set +e

python example_module/import_test.py

cp example_module/import_test.py .
python import_test.py
