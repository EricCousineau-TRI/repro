from ctypes import *

# Use PYFUNCTYPE so we maintain GIL, so that MATLAB doesn't huff, and puff,
# and crash with a segfault.

# py_raw_t (mx_raw_t mx_raw_handle, int nargout, py_raw_t py_raw_in)
mx_raw_t = c_uint64
py_raw_t = c_uint64
c_mx_feval_py_raw_t = PYFUNCTYPE(py_raw_t, mx_raw_t, c_int, py_raw_t)

# Simple example.
c_simple_t = PYFUNCTYPE(c_int)

# py - Python
# py_raw - Raw Python points (py_object, c_void_p)
# mx - MATLAB / mxArray*
# mx_raw - void* representing mxArray* (mx_array_t, c_void_p)
# mex - MEX function

class MxRaw:
    def __init__(self, value, name):
        self.value = value
        self.name = name
    def __str__(self):
        return "<MxRaw: {}>".format(self.name)

# MATLAB Function handle
class MxFunc(MxRaw):
    def __init__(self, value, name):
        MxRaw.__init__(self, value, name)
    def __str__(self):
        return "<MxFunc: {}>".format(self.name)
    def call(self, args, nargout=1, unpack_scalar=True):
        out = mx_raw_feval_py(self.value, nargout, *args)
        if len(out) == 1 and unpack_scalar:
            out = out[0]
        return out
    def __call__(self, *args, **kwargs):
        return self.call(args, **kwargs)

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
    # Calling MEX through type erasure.
    funcs['c_mx_feval_py_raw'] = \
        c_mx_feval_py_raw_t(funcs_in['c_mx_feval_py_raw'])
    funcs['c_simple'] = \
        c_simple_t(funcs_in['c_simple'])

def simple():
    funcs['c_simple']()

# Used by MATLAB 
# TODO: Consider having similar Erasure mechanism, since MATLAB is not pointer-friendly.
def py_raw_to_py(py_raw):
    # py_raw - will be uint64
    py = erasure.dereference(py_raw)
    return py

def py_to_py_raw(py):
    py_raw = erasure.store(py)
    return py_raw

# Used by Python
def mx_raw_feval_py(mx_raw_handle, nargout, *py_in):
    # Just do a py.list, for MATLAB to convert to a cell arrays.
    mx_feval_py_raw = funcs['c_mx_feval_py_raw']
    py_raw_in = py_to_py_raw(py_in)
    try:
        py_raw_out = (mx_feval_py_raw(
            c_uint64(mx_raw_handle), c_int(int(nargout)), c_uint64(py_raw_in)))
        py_out = py_raw_to_py(py_raw_out)
    except:
        import sys, traceback
        print "py: error"
        traceback.print_exc(file=sys.stdout)
        py_out = None
    return py_out

if __name__ == "__main__":
    # Test erasure
    i1 = erasure.store(1)
    i2 = erasure.store({"hello": 1})
    assert(erasure.dereference(i1) == 1)
    assert(erasure.dereference(i2) == {"hello": 1})
