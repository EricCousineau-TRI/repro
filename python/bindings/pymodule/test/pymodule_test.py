#!/usr/bin/env python
from __future__ import print_function, absolute_import

# Ensure that we can import the whole module.

import unittest
import pymodule
from pymodule import inherit_check as ic
from pymodule import type_binding as tb

class TestTypeBinding(unittest.TestCase):
    def test_type_binding(self):
        obj = tb.SimpleType(1)
        self.assertTrue(obj is not None)
    
    def test_inherit_check(self):
        obj = ic.CppExtend()
        self.assertTrue(obj is not None)

if __name__ == '__main__':
    unittest.main()
