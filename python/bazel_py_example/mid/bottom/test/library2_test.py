import bazel_py_example.mid.bottom.library2 as mut

import unittest


class TestLibrary2(unittest.TestCase):
    def test_library2_func(self):
        self.assertEqual(mut.library2_func(), 2)


if __name__ == "__main__":
    unittest.main()
