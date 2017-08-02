#!/usr/bin/env python
from __future__ import print_function, absolute_import

import unittest
from pymodule.global_check.consumer_1 import do_stuff_1
from pymodule.global_check.consumer_2 import do_stuff_2

class TestInheritance(unittest.TestCase):
    def test_basic(self):
        value = 2
        print(do_stuff_1(value))
        print(do_stuff_2(value))

if __name__ == '__main__':
    unittest.main()
