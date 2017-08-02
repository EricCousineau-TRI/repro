#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest
import numpy as np
npa = np.array
import pymodule
from pymodule.sub import inherit_check as ic

class PyExtend(ic.Base):
    def pure(self, value):
        return "py.pure=" + value
    def optional(self, value):
        return "py.optional=" + value

class TestInheritance(unittest.TestCase):
    def test_basic(self):
        cpp = ic.CppExtend()
        py = PyExtend()
        value = "a"
        self.assertEqual(cpp.dispatch(value), "cpp.dispatch: cpp.pure=a cpp.optional=a")
        self.assertEqual(py.dispatch(value), "cpp.dispatch: py.pure=a py.optional=a")

if __name__ == '__main__':
    unittest.main()
