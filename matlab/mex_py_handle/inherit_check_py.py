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
    def __init__(self, mx_obj, mx_feval):  
        self.mx_obj = mx_obj
        self.mx_feval = mx_feval
    def pure(self, value):
        print "py: pyInvoke pure - start"
        out = self.mx_feval('pyInvoke', 'pure', self.mx_obj, value)
        print "py: pyInvoke pure - finish"
        return out
    def optional(self, value):
        print "py: pyInvoke optional - start"
        out = self.mx_feval('pyInvoke', 'optional', self.mx_obj, value)
        print "py: pyInvoke optional - finish"
        return out
    def roundabout_dispatch(self, value):
        print value
        print type(value)
        return self.dispatch(value)
