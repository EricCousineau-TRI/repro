from pymodule.sub import inherit_check as ic

class PyExtend(ic.Base):
    def pure(self, value):
        print("py.pure={}".format(value))
        return value
    def optional(self, value):
        print("py.optional={}".format(value))
        return value * 100
