from __future__ import absolute_import, print_function
import py_mex_proxy

from pymodule.sub import inherit_check as ic

class PyExtend(ic.Base):
    def __init__(self):
        super(PyExtend, self).__init__()
    def pure(self, value):
        print("py.pure={}".format(value))
        return value
    def optional(self, value):
        print("py.optional={}".format(value))
        return value * 100

# Example Trampoline class
class PyMxExtend(py_mex_proxy.PyMxClass(PyExtend)):
    def __init__(self, mx_obj):
        super(PyMxExtend, self).__init__(mx_obj=mx_obj)
        self._mx_decl_virtual('pure')
        self._mx_decl_virtual('optional')
    # # Alternative: Manually declaring each.
    # def pure(self, value):
    #     return self._mx_virtual('pure', value)
    # def optional(self, value):
    #     return self._mx_virtual('optional', value)

if __name__ == "__main__":
    dut = PyMxExtend(1)
    print(dut)
