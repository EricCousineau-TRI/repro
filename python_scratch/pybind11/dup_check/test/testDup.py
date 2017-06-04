#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest

# Goal: See if Python complains about dynamic modules having the same relative name.
# Result: Seems fine.

class TestDup(unittest.TestCase):
    def import_modules_with_same_rel_name(self):
        from pydrake import _dup as d1
        from pydrake.sub import _dup as d2
        self.assertEqual(d1.get_name(), "root level")
        self.assertEqual(d2.get_name(), "sub level")

if __name__ == '__main__':
    unittest.main()
