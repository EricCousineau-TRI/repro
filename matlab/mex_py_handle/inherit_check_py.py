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

class PyMxExtend(ic.Base):
    # Cheat with mx_feval for now, for simplicity
    def __init__(self, mx_obj, mx_obj_feval_mx_raw):
        super(PyMxExtend, self).__init__()
        self.mx_obj = mx_obj
        self.mx_feval = mx_obj_feval_mx_raw
    def pure(self, value):
        print "py: pyInvoke pure - start"
        out = self.mx_feval(self.mx_obj, 'pyInvokeVirtual', 'pure', value)
        print "py: pyInvoke pure - finish"
        return out
    def optional(self, value):
        print "py: pyInvoke optional - start"
        out = self.mx_feval(self.mx_obj, 'pyInvokeVirtual', 'optional', value)
        print "py: pyInvoke optional - finish"
        return out
    def roundabout_dispatch(self, value):
        print "py: roundabout_dispatch - start"
        print "  {}".format(self.mx_obj)
        print value
        print type(value)
        out = self.dispatch(value)
        print "py: roundabout_dispatch - end"
        return out
