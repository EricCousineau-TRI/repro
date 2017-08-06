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
        self.mx_obj_feval_mx_raw = mx_obj_feval_mx_raw
    def _mx_virtual(self, method, *args):
        return self.mx_obj_feval_mx_raw(
            self.mx_obj, 'pyInvokeVirtual', method, *args)
    def free(self):
        print "py: free"
        self.mx_obj.free()
        self.mx_obj = None
        self.mx_obj_feval_mx_raw.free()
        self.mx_obj_feval_mx_raw = None
    def pure(self, value):
        return self._mx_virtual('pure', value)
    def optional(self, value):
        return self._mx_virtual('optional', value)
    # def dispatch(self, value):
    #     # print "Sidetrack: {}".format(value)
    #     # return self.pure(value) + self.optional(value)
    #     print "py: dispatch"
    #     # return 1
    #     return ic.Base.dispatch(self, value)
