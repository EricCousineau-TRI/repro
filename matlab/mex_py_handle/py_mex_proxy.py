from ctypes import *

# py_raw_t (mx_raw_t mx_raw_handle, int nout, py_raw_t py_raw_in)
# - Use PYFUNCTYPE so we maintain GIL (so that MATLAB doesn't get huffy and puffy about it).
c_mx_feval_py_raw_t = PYFUNCTYPE(c_uint64, c_uint64, c_int, c_uint64)

c_simple_t = PYFUNCTYPE(c_int)

# void* (void*) - but use ctypes to extract void* from py_object
c_py_raw_to_py_t = PYFUNCTYPE(py_object, c_void_p)  # Use py_object so ctypes can cast to void*
c_py_to_py_raw_t = PYFUNCTYPE(c_void_p, py_object)

# py - Python
# py_raw - Raw Python points (py_object, c_void_p)
# mx (ml) - MATLAB / mxArray*
# mx_raw - void* representing mxArray* (mx_array_t, c_void_p)
# mex - MEX function

class Erasure(object):
    def __init__(self):
        self._values = []
        self._occupied = []
    def store(self, value):
        i = -1
        for (i, occ) in enumerate(self._occupied):
            if not occ:
                break
        if i == -1:
            i = self._size()
            self._resize(self._size() + 4)
        assert(self._values[i] is None)
        assert(not self._occupied[i])
        self._values[i] = value
        self._occupied[i] = True
        return i
    def dereference(self, i, keep=False):
        assert(i < self._size())
        assert(self._occupied[i])
        value = self._values[i]
        if not keep:
            self._values[i] = None
            self._occupied[i] = False
        return value
    def _size(self):
        return len(self._values)
    def _resize(self, new_sz):        
        assert(new_sz >= self._size())
        dsz = new_sz - self._size()
        self._values += [None] * dsz
        self._occupied += [False] * dsz


funcs = {}
erasure = Erasure()

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
    funcs['c_simple'] = \
        c_simple_t(funcs_in['c_simple'])
    print "py: stored pointers"

def simple():
    print "py: c_simple - start"
    funcs['c_simple']()
    print "py: c_simple - finish"

# Used by MATLAB 
# TODO: Consider having similar Erasure mechanism, since MATLAB is not pointer-friendly.
def py_raw_to_py(py_raw):
    # py_raw - will be uint64
    # py = funcs['c_py_raw_to_py'](py_raw)
    py = erasure.dereference(py_raw)
    return py

def py_to_py_raw(py):
    # py_raw = funcs['c_py_to_py_raw'](py_object(py))
    py_raw = erasure.store(py)
    return py_raw

# Used by Python
def mx_raw_feval_py(mx_raw_handle, nout, *py_in):
    simple()
    # Just do a py.list, for MATLAB to convert to a cell arrays.
    # py_raw_in = py_to_py_raw(py_in)
    print "py: mx_raw_feval_py - start"
    mx_feval_py_raw = funcs['c_mx_feval_py_raw']
    py_out = None
    try:
        py_raw_in = py_to_py_raw(py_in)
        py_raw_out = (mx_feval_py_raw(
                c_uint64(mx_raw_handle), c_int(int(nout)), c_uint64(py_raw_in)))
        py_out = py_raw_to_py(py_raw_out)
    except:
        import sys, traceback
        print "py: error"
        traceback.print_exc(file=sys.stdout)
    print "py: mx_raw_feval_py - finish"
    return py_out

if __name__ == "__main__":
    # Test erasure
    i1 = erasure.store(1)
    i2 = erasure.store({"hello": 1})
    assert(erasure.dereference(i1) == 1)
    assert(erasure.dereference(i2) == {"hello": 1})
