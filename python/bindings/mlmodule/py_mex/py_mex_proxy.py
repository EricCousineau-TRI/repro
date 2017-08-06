#!/usr/bin/env python
from __future__ import absolute_import, print_function

import traceback
from ctypes import PYFUNCTYPE, c_uint64, c_int

from util import Erasure

debug = False

# py - Python
# py_raw - Raw Python points (py_object, c_void_p)
# mx - MATLAB / mxArray*
# mx_raw - void* representing mxArray* (mx_array_t, c_void_p)
# mex - MEX function

# Use PYFUNCTYPE so we maintain GIL, so that MATLAB doesn't huff, and puff,
# and crash with a segfault.

# py_raw_t (mx_raw_t mx_raw_handle, int nargout, py_raw_t py_raw_in)
mx_raw_t = c_uint64
# TODO: For py_raw_t, consider using py_object in lieu of c_uint64.
# (No need for .util.Erasure, then.)
py_raw_t = c_uint64
c_mx_feval_py_raw_t = PYFUNCTYPE(py_raw_t, mx_raw_t, c_int, py_raw_t)
c_mx_raw_ref_t = PYFUNCTYPE(c_int, mx_raw_t)

# Simple example.
c_simple_t = PYFUNCTYPE(c_int)

# Globals
mx_funcs = {}
funcs = {}

def free():
    global funcs
    global mx_funcs
    funcs = {}
    mx_funcs = {}

# For obtaining function pointers from a MEX function.
def init_c_func_ptrs(funcs_in):
    global funcs
    # Calling MEX through type erasure.
    funcs['c_mx_feval_py_raw'] = \
        c_mx_feval_py_raw_t(funcs_in['c_mx_feval_py_raw'])
    funcs['c_simple'] = \
        c_simple_t(funcs_in['c_simple'])
    funcs['c_mx_raw_ref_incr'] = \
        c_mx_raw_ref_t(funcs_in['c_mx_raw_ref_incr'])
    funcs['c_mx_raw_ref_decr'] = \
        c_mx_raw_ref_t(funcs_in['c_mx_raw_ref_decr'])

def init_mx_funcs(mx_funcs_in):
    global mx_funcs
    mx_funcs = mx_funcs_in

# Test function
def simple():
    funcs['c_simple']()

def mx_feval(*args, **kwargs):
    feval = mx_funcs['feval']
    # See MxFunc signature.
    return feval(*args, **kwargs)

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
# Marhsal types to opaque, C-friendly types, that will then be passed
# to MATLAB via `MexPyProxy.mx_feval_py_raw`.
def mx_raw_feval_py(mx_raw_handle, nargout, *py_in):
    mx_feval_py_raw = funcs['c_mx_feval_py_raw']
    py_raw_in = py_to_py_raw(py_in)
    if debug:
        print("py.erasure: {}".format(erasure._values))
    py_raw_out = (mx_feval_py_raw(
        c_uint64(mx_raw_handle), c_int(int(nargout)), c_uint64(py_raw_in)))
    if py_raw_out == 0xBADF00D:
        traceback.print_stack()
        raise Exception("Error")
    py_out = py_raw_to_py(py_raw_out)
    if debug:
        print("py.erasure: {}".format(erasure._values))
    return py_out

def mx_raw_ref_incr(mx_raw):
    out = funcs['c_mx_raw_ref_incr'](mx_raw)
    if out != 0:
        traceback.print_stack()
        raise Exception("Error")

def mx_raw_ref_decr(mx_raw):
    out = funcs['c_mx_raw_ref_decr'](mx_raw)
    if out != 0:
        traceback.print_stack()
        raise Exception("Error")

# Wrap a raw MATLAB type, and use referencing counting to tie it to the lifetime
# of this object.
class MxRaw(object):
    def __init__(self, mx_raw, disp):
        self.mx_raw = mx_raw
        self.disp = disp
        mx_raw_ref_incr(self.mx_raw)
        if debug:
            print("py: Store {}".format(self))
    def free(self):
        if self.mx_raw is not None:
            mx_raw_ref_decr(self.mx_raw)
            self.mx_raw = None
    def __del__(self):
        if debug:
            print("py: Destroy {}".format(self))
        self.free()
    def __str__(self):
        return "<MxRaw: {}>".format(self.disp)

# MATLAB Function handle
class MxFunc(MxRaw):
    def __init__(self, mx_raw, disp):
        super(MxFunc, self).__init__(mx_raw, disp)

    def __str__(self):
        return "<MxFunc: {}>".format(self.disp)

    def call(self, args, nargout=1, unpack_scalar=True):
        if self.mx_raw is None:
            raise Exception("Already destroyed")
        out = mx_raw_feval_py(self.mx_raw, nargout, *args)
        if len(out) == 1 and unpack_scalar:
            out = out[0]
        return out
    def __call__(self, *args, **kwargs):
        return self.call(args, **kwargs)

def is_trampoline(obj):
    return hasattr(obj, 'mx_obj')

# Type inheritance composition.
# (Multiple inheritance is an unwanted beast at this moment.)
def PyMxClass(BaseCls):
    class PyMxClassImpl(BaseCls):
        def __init__(self, mx_obj, *args, **kwargs):
            super(PyMxClassImpl, self).__init__(*args, **kwargs)
            # `mx_obj` should be a `MxRaw`
            # TODO: This should be a weak reference to the MATLAB object...
            # but there doesn't seem to be a way to do that...
            assert mx_obj is not None
            self._mx_obj = mx_obj

        def _mx_virtual(self, method, *args):
            assert self._mx_obj is not None
            return mx_feval('pyInvokeVirtual', self._mx_obj, method, *args)

        def _mx_free(self):
            # Explicitly permit free'ing due to cyclic references... and inability
            # to access MATLAB's reference counting...
            self._mx_obj.free()
            self._mx_obj = None

        def _mx_decl_virtual(self, method):
            # Add method
            def func(*args):
                return self._mx_virtual(method, *args)
            self.__dict__[method] = func

    return PyMxClassImpl
