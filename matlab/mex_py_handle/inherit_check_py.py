from pymodule.sub import inherit_check as ic

import py_mex_proxy

class PyExtend(ic.Base):
    def __init__(self, func):
        super(PyExtend, self).__init__()
        self.func = func
    def pure(self, value):
        print("py.pure={}".format(value))
        self.func(value)
        return value
    def optional(self, value):
        print("py.optional={}".format(value))
        self.func(value)
        return value * 100

# Composition.
def PyMxClass(BaseCls):
    class PyMxClassImpl(BaseCls):
        def __init__(self, mx_obj, *args, **kwargs):
            super(PyMxClassImpl, self).__init__(*args, **kwargs)
            # `mx_obj` should be a `py_mex_proxy.MxRaw`
            # TODO: This should be a weak reference to the MATLAB object...
            # but there doesn't seem to be a way to do that...
            assert mx_obj is not None
            self._mx_obj = mx_obj

        def _mx_virtual(self, method, *args):
            assert self._mx_obj is not None
            return py_mex_proxy.mx_feval('pyInvokeVirtual', self._mx_obj, method, *args)

        def _mx_free(self):
            # Explicitly permit free'ing due to cyclic references... and inability
            # to access MATLAB's reference counting...
            self._mx_obj.free()
            self._mx_obj = None

        def _mx_decl(self, method):
            # Add method
            def func(*args):
                return self._mx_virtual(method, *args)
            self.__dict__[method] = func

    return PyMxClassImpl

# Example Trampoline class
class PyMxExtend(PyMxClass(ic.Base)):
    # How to handle different constructor arguments with multiple inheritance?
    def __init__(self, mx_obj):
        super(PyMxExtend, self).__init__(mx_obj=mx_obj)
        self._mx_decl('pure')
        self._mx_decl('optional')
    # def pure(self, value):
    #     return self._mx_virtual('pure', value)
    # def optional(self, value):
    #     return self._mx_virtual('optional', value)

if __name__ == "__main__":
    c = PyMxExtend(1)
    print(c)
