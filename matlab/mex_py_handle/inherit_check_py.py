from pymodule.sub import inherit_check as ic

class PyExtend(ic.Base):
    def pure(self, value):
        return "py.pure=" + value
    def optional(self, value):
        return "py.optional=" + value

class PyMxExtend(ic.Base):
    # Cheat with mx_feval for now, for simplicity
    def __init__(self, mx_obj, mx_feval):  
        self.mx_obj = mx_obj
        self.mx_feval = mx_feval
    def pure(self, value):
        return self.mx_feval('pyInvoke', 'pure', self.mx_obj)
    def optional(self, value):
        return self.mx_feval('pyInvoke', 'optional', self.mx_obj)
