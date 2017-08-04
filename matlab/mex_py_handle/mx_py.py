from ctypes import *

# int (void* mx_raw_handle, int nout, void* py_raw_in)
c_mx_feval_py_raw_t = CFUNCTYPE(c_void_p, c_int_p, c_void_p)

# void* (void*) - but use ctypes to extract void* from py_object
c_raw_to_py_t = ctypes.CFUNCTYPE(py_object, c_void_p)  # Use py_object so ctypes can cast to void*
c_py_to_raw_t = ctypes.CFUNCTYPE(c_void_p, py_object)

# py - Python
# py_raw - Raw Python points (py_object, c_void_p)
# mx (ml) - MATLAB / mxArray*
# mx_raw - void* representing mxArray* (mx_array_t, c_void_p)
# mex - MEX function

funcs = {}

def init_c_func_ptrs(funcs_in):
    global funcs
    # Effectively re-interperet casts.
    funcs['c_raw_to_py'] = \
        c_raw_to_py_t(funcs_in['c_raw_to_py'])
    funcs['c_py_to_raw'] = \
        c_py_to_raw_t(funcs_in['c_py_to_raw'])
    # Calling MEX through type erasure.
    funcs['c_mx_feval_py_raw'] = \
        c_mx_feval_py_raw_t(funcs_in['c_mx_feval_py_raw'])

def py_raw_to_py(py_raw):
    return funcs['c_raw_to_py'](py_raw)

def py_to_py_raw(obj):
    return funcs['c_py_to_raw'](obj)

def mx_feval_py(mx_raw_handle, nout, *py_in):
    nargin = len(py_in)
    # Just do a py.list, for MATLAB to convert to a cell arrays.
    py_raw_in = py_to_py_raw(py_in)
    mx_feval_py_raw = funcs['c_mx_feval_py_raw']
    py_raw_out = mx_feval_py_raw(mx_raw_handle, nout, py_raw_in)
    py_out = py_raw_to_py(nout, py_raw_out)
    return py_out
