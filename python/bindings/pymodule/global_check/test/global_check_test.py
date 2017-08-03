#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest

from pymodule.util.share_symbols import ShareSymbols
with ShareSymbols():
    import pymodule.global_check.consumer_1 as c1
    import pymodule.global_check.consumer_2 as c2

class TestInheritance(unittest.TestCase):
    def test_basic(self):
        value = 2
        # Will have the same singletons.
        for i in xrange(3):
            (ptr1, value1) = c1.consume(value)
            (ptr2, value2) = c2.consume(value)
            self.assertEqual(ptr1, ptr2)
            self.assertEqual(value1 + value, value2)
        print(ptr1, value1, value2)

if __name__ == '__main__':
    unittest.main()
