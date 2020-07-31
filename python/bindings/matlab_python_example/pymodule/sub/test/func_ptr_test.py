#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest
from pymodule.sub import func_ptr as fp

class TestInheritance(unittest.TestCase):
    def test_basic(self):
        # https://stackoverflow.com/questions/4851463/python-closure-write-to-variable-in-parent-scope
        value = [1]
        def tmp(x):
            value[0] += x
            return value[0]
        out = fp.call_cpp(tmp)
        self.assertEqual(2, out)
        self.assertEqual(2, value[0])
        print(value[0])

if __name__ == '__main__':
    unittest.main()
