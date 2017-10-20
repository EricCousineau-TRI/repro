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
        (ptr1, value1) = c1.consume_linkstatic(value)
        (ptr2, value2) = c2.consume_linkstatic(value)
        self.assertEqual(ptr1, ptr2)
        self.assertEqual(value1 + value, value2)
        print("linkstatic\n{}\n{}\n{} {}\n".format(ptr1, ptr2, value1, value2))

        (ptr1b, value1b) = c1.consume_linkshared(value)
        (ptr2b, value2b) = c2.consume_linkshared(value)
        self.assertEqual(ptr1b, ptr2b)
        self.assertEqual(value1b + value, value2b)
        print("linkshared\n{}\n{}\n{} {}\n".format(ptr1b, ptr2b, value1b, value2b))


if __name__ == '__main__':
    unittest.main()
