from __future__ import print_function, absolute_import

import unittest
import numpy as np
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
        err_expected = RuntimeError
        bad_ctor = lambda: tb.SimpleType(1.5)
        self.assertRaises(err_expected, bad_ctor)

if __name__ == '__main__':
    unittest.main()
