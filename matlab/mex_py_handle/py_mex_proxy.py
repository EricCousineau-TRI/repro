#!/usr/bin/env python
from ctypes import *

from util import Erasure

# py - Python
# py_raw - Raw Python points (py_object, c_void_p)
# mx - MATLAB / mxArray*
# mx_raw - void* representing mxArray* (mx_array_t, c_void_p)
# mex - MEX function

# Use PYFUNCTYPE so we maintain GIL, so that MATLAB doesn't huff, and puff,
# and crash with a segfault.

# py_raw_t (mx_raw_t mx_raw_handle, int nargout, py_raw_t py_raw_in)
mx_raw_t = c_uint64
py_raw_t = c_uint64
c_mx_feval_py_raw_t = PYFUNCTYPE(py_raw_t, mx_raw_t, c_int, py_raw_t)

# Simple example.
c_simple_t = PYFUNCTYPE(c_int)

funcs = {}
# For obtaining function pointers from a MEX function.
def init_c_func_ptrs(funcs_in):
    global funcs
    # Calling MEX through type erasure.
    funcs['c_mx_feval_py_raw'] = \
        c_mx_feval_py_raw_t(funcs_in['c_mx_feval_py_raw'])
    funcs['c_simple'] = \
        c_simple_t(funcs_in['c_simple'])

# Test function
def simple():
    funcs['c_simple']()

# Used by MATLAB 
# TODO: Consider returning to using ctypes.
erasure = Erasure()
def py_raw_to_py(py_raw):
    # py_raw - will be uint64
    py = erasure.dereference(py_raw)
    return py

def py_to_py_raw(py):
    py_raw = erasure.store(py)
    return py_raw

# Used by Python
def mx_raw_feval_py(mx_raw_handle, nargout, *py_in):
    # Marhsal types to opaque, C-friendly types, that will then be passed
    # to MATLAB via `MexPyProxy.mx_feval_py_raw`.
    mx_feval_py_raw = funcs['c_mx_feval_py_raw']
    py_raw_in = py_to_py_raw(py_in)
    py_raw_out = (mx_feval_py_raw(
        c_uint64(mx_raw_handle), c_int(int(nargout)), c_uint64(py_raw_in)))
    py_out = py_raw_to_py(py_raw_out)
    return py_out

# Wrap a raw MATLAB type.
class MxRaw:
    def __init__(self, value, name):
        self.value = value
        self.name = name
    def __str__(self):
        return "<MxRaw: {}>".format(self.name)

# MATLAB Function handle
class MxFunc:
    def __init__(self, value, name):
        # MxRaw.__init__(self, value, name)
        self.name = name
        self.value = value
    def __str__(self):
        return "<MxFunc: {}>".format(self.name)
    def call(self, args, nargout=1, unpack_scalar=True):
        out = mx_raw_feval_py(self.value, nargout, *args)
        if len(out) == 1 and unpack_scalar:
            out = out[0]
        return out
    def __call__(self, *args, **kwargs):
        return self.call(args, **kwargs)
