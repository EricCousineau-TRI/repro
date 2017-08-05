from pymodule.sub import inherit_check as ic

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

# Example Trampoline class
class PyMxExtend(ic.Base):
    # Cheat with mx_feval for now, for simplicity
    def __init__(self, mx_obj, mx_obj_feval_mx_raw):
        super(PyMxExtend, self).__init__()
        # This should be a `py_mex_proxy.MxRaw`
        self.mx_obj = mx_obj
        def mx_virtual(method, *args):
            mx_obj_feval_mx_raw(mx_obj, 'pyInvokeVirtual', method, *args)
        self.mx_virtual = mx_virtual
    def pure(self, value):
        return self.mx_virtual('pure', value)
    def optional(self, value):
        return self.mx_virtual('optional', value)
