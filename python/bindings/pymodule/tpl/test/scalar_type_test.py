#!/usr/bin/env python
from __future__ import print_function, absolute_import

# import unittest
# import pymodule
from pymodule.tpl import scalar_type as st

print("\n".join(sorted(st.__dict__.keys())))
print(st.Base__T_double__U_int.type_tup)
print(st.Base__T_int__U_double.type_tup)

# class PyExtend(ic.Base):
#     def pure(self, value):
#         print("py.pure={}".format(value))
#         return value
#     def optional(self, value):
#         print("py.optional={}".format(value))
#         return value * 100

# class TestInheritance(unittest.TestCase):
#     def test_basic(self):
#         cpp = ic.CppExtend()
#         py = PyExtend()
#         value = 2
#         self.assertEqual(cpp.dispatch(value), 22)
#         self.assertEqual(py.dispatch(value), 202)

# if __name__ == '__main__':
#     unittest.main()
