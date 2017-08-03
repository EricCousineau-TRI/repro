#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest

# This not enable symbol sharing, and will fail.
import pymodule.global_check.consumer_1 as c1
import pymodule.global_check.consumer_2 as c2

class TestInheritance(unittest.TestCase):
    def test_basic(self):
        value = 2
        print(c1.do_stuff_1(value))
        print(c2.do_stuff_2(value))

if __name__ == '__main__':
    unittest.main()
