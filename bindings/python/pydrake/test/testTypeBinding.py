#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest
import numpy as np
npa = np.array
import pydrake
from pydrake import typebinding as tb

class TestTypeBinding(unittest.TestCase):
    def test_env(self):
        import os
        os.system('echo "PATH=$PATH\n\nPYTHONPATH=$PYTHONPATH\n\nLD_LIBRARY_PATH=$LD_LIBRARY_PATH"')

    def test_basic(self):
        obj = tb.SimpleType(1)
        self.assertEqual(obj.value(), 1)
        obj.set_value(2)
        self.assertEqual(obj.value(), 2)

    def test_flexible(self):
        obj = tb.SimpleType(1.)
        self.assertEqual(obj.value(), 1)
        obj.set_value(2.)
        self.assertEqual(obj.value(), 2)
        # Expect non-integral floating point values to throw error
        bad_ctor = lambda: tb.SimpleType(1.5)
        self.assertRaises(RuntimeError, bad_ctor)
        bad_set = lambda: obj.set_value(1.5)
        self.assertRaises(RuntimeError, bad_set)
        bad_type = lambda: obj.set_value("bad")
        self.assertRaises(TypeError, bad_type)

    def test_numpy_basic(self):
        # Will reshape a flat nparray. Need to explicitly shape.
        value = npa([[1., 2, 3]]).T
        obj = tb.EigenType(value)
        self.assertTrue(np.allclose(obj.value(), value))

        value_flat = npa([1, 2, 3])
        obj.set_value(value_flat)
        self.assertTrue(np.allclose(obj.value(), value))

    def test_numpy_flexible(self):
        # Set scalars, and let these be freely interpreted as matrices.
        obj = tb.EigenType(1.)
        self.assertTrue(np.allclose(obj.value(), 1.))
        obj.set_value(2.)
        self.assertEqual(obj.value(), 2.)
        # Hmm... Was not expecting this to work... Will need to figure out why this happens.
        obj.set_value(1)
        self.assertEqual(obj.value(), 1)

if __name__ == '__main__':
    unittest.main()
