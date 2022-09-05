import unittest

import example.common.cc as mut


class Test(unittest.TestCase):
    def test_basic(self):
        print(mut.FuncA())
        print(mut.FuncB())


if __name__ == "__main__":
    unittest.main()
