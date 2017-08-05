from pymodule.sub import inherit_check as ic

class PyExtend(ic.Base):
    def pure(self, value):
        return "py.pure=" + value
    def optional(self, value):
        return "py.optional=" + value
