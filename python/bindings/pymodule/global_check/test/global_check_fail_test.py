#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest

import pymodule.global_check.consumer_1 as c1
import pymodule.global_check.consumer_2 as c2

class TestInheritance(unittest.TestCase):
    def test_basic(self):
        value = 2
        # Will have separate singletons.
        (ptr1, value1) = c1.consume(value)
        (ptr2, value2) = c2.consume(value)
        self.assertNotEqual(ptr1, ptr2)
        self.assertEqual(value1, value2)

if __name__ == '__main__':
    unittest.main()
