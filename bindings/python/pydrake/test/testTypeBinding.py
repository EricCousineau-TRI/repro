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

    def test_flexible(self):
        """
        Presently fails:
        
            TypeError: __init__(): incompatible constructor arguments. The following argument types are supported:
                1. pydrake._pydrake_typebinding.SimpleType()
                2. pydrake._pydrake_typebinding.SimpleType(arg0: int)
        """
        obj = tb.SimpleType(1.)
        self.assertEqual(obj.value(), 1)

if __name__ == '__main__':
    unittest.main()
