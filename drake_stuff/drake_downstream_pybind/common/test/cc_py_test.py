import unittest

import example.common.cc as mut


class Test(unittest.TestCase):
    def test_basic(self):
        mut.FuncA()
        mut.FuncB()


if __name__ == "__main__":
    unittest.main()
