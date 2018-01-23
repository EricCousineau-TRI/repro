import unittest

from cpp_const import ConstError

import test._cpp_const_pybind_test_py as m

class TestCppConstPybind(unittest.TestCase):
    def ex(self):
        return self.assertRaises(TypeError)

    def ex_const(self):
        # unittest does not generalize on assertion errors? :(
        return self.assertRaises(ConstError)

    def test_basics(self):
        obj = m.Test(10)
        obj_const = obj.as_const()
        self.assertEquals(obj_const.value, 10)
        obj.value = 100
        self.assertEquals(obj_const.value, 100)

        m.func_const(obj)
        m.func_mutate(obj)
        obj.check_const()
        obj.check_mutate()

        m.func_const_extra(obj_const, "hello")
        obj_const.check_const()
        obj_const.check_mutate()
        with self.ex(): obj_const.check_mutate()
        with self.ex(): m.func_mutate(obj_const)
        with self.ex_const(): obj_const.value = 1000

if __name__ == "__main__":
    unittest.main()
