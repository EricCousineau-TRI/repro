#!/usr/bin/env python
import unittest

import cpp_const as m



# Not annotated.
class Basic(object):
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    name = property(get_name, set_name)


class TestCppConst(unittest.TestCase):
    def ex(self):
        return self.assertRaises(m.ConstError)

    def test_list(self):
        # List.
        x = [1, 2, 3, [10]]
        x_const = m.to_const(x)
        with self.ex(): x_const[0] = 10
        with self.ex(): x_const[:] = []
        with self.ex(): del x_const[0]
        with self.ex(): x_const.append(10)
        with self.ex(): x_const.pop()
        # Test iteration.
        for i_const in x_const:
            self.assertTrue(m.is_const_or_immutable(i_const))
        # N.B. Access does not propagate...
        self.assertTrue(isinstance(x_const[3], list))
        x_const[3].pop()

    def test_dict(self):
        d = {"a": 0, "b": 1, "z": [25]}
        d_const = m.to_const(d)
        self.assertEquals(d_const["a"], 0)
        with self.ex(): d_const["c"] = 2
        with self.ex(): d_const.clear()
        # N.B. Access does not propagate.
        d_const["z"].pop()

    def test_basic(self):
        obj = Basic("Tim")
        obj_const = m.to_const(obj)
        self.assertEquals(obj_const.get_name(), "Tim")
        with self.ex(): obj_const.set_name("Bob")
        with self.ex(): obj_const.name = "Bob"
        with self.ex(): obj_const._name = "Bob"


if __name__ == "__main__":
    unittest.main()
