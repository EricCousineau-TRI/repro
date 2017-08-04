from ctypes import *

# void* (mx_raw_t mx_raw_handle, int nout, void* py_raw_in)
c_mx_feval_py_raw_t = CFUNCTYPE(py_object, c_uint64, c_int, py_object)

# void* (void*) - but use ctypes to extract void* from py_object
c_py_raw_to_py_t = CFUNCTYPE(py_object, c_void_p)  # Use py_object so ctypes can cast to void*
c_py_to_py_raw_t = CFUNCTYPE(c_void_p, py_object)

# py - Python
# py_raw - Raw Python points (py_object, c_void_p)
# mx (ml) - MATLAB / mxArray*
# mx_raw - void* representing mxArray* (mx_array_t, c_void_p)
# mex - MEX function

funcs = {}

def init_c_func_ptrs(funcs_in):
    global funcs
    # Effectively re-interperet casts.
    funcs['c_py_raw_to_py'] = \
        c_py_raw_to_py_t(funcs_in['c_py_raw_to_py'])
    funcs['c_py_to_py_raw'] = \
        c_py_to_py_raw_t(funcs_in['c_py_to_py_raw'])
    # Calling MEX through type erasure.
    funcs['c_mx_feval_py_raw'] = \
        c_mx_feval_py_raw_t(funcs_in['c_mx_feval_py_raw'])
    print "Stored pointers"

# Used by MATLAB
# TODO: Consider having similar Erasure mechanism, since MATLAB is not pointer-friendly.
def py_raw_to_py(py_raw):
    # py_raw - will be uint64
    py = funcs['c_py_raw_to_py'](py_raw)
    return py

def py_to_py_raw(py):
    py_raw = funcs['c_py_to_py_raw'](py)
    return py_raw

# Used by Python
def mx_raw_feval_py(mx_raw_handle, nout, *py_in):
    print "Calling from Python"
    # Just do a py.list, for MATLAB to convert to a cell arrays.
    # py_raw_in = py_to_py_raw(py_in)
    mx_feval_py_raw = funcs['c_mx_feval_py_raw']
    try:
        print (mx_raw_handle, int(nout), py_in)
        py_out = mx_feval_py_raw(c_uint64(mx_raw_handle), c_int(int(nout)), py_object(py_in))
    except:
        import sys, traceback
        print "Error"
        traceback.print_exc(file=sys.stdout)
        py_out = None
    # py_out = py_raw_to_py(nout, py_raw_out)
    return py_out
