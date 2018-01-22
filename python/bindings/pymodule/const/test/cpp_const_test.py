#!/usr/bin/env python
import unittest

from cpp_const_test import const_meta, mutable_method, to_const, to_mutable



# Not annotated.
class Basic(obect):
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    name = property(get_name, set_name)

# Annotate owned properties that may have levels of indirection involved.
@const_meta(owned_properties = ['_map'])
class Base(object):
    def __init__(self, value):
        self._value = value
        self._map = {10: 0}

    def print_self(self):
        print(type(self), self)

    def set_value(self, value):
        self._value = value

    def get_map(self, key):
        return self._map[key]

    def set_map(self, key, value):
        self._map[key] = value

    def do_something(self, stuff):
        print("{}: {}".format(stuff, self._value))
        # self.set_value(10)  # Raises error.

    @mutable_method
    def mutate(self): pass

    def __setitem__(self, key, value):
        self._map[key] = value

    value = property(get_value, set_value)


@const_meta(owned_properties = ['_my_value'])
class SubCheck(Check):
    def __init__(self):
        Check.__init__(self, 100)
        self._my_value = []

    def extra(self):
        self.set_map(10, 10000)

    def more(self):
        # self._my_value.append(10)  # Raises error.
        return self._my_value


c = Check(10)
c_const = to_const(c)
print(c_const.value)
# c_const.value = 100  # Raises error.
# c_const.set_value(100)  # Raises error.
c_const.do_something("yo")
print(c_const == c_const)
print(c_const.get_map(10))
# print(c_const.set_map(10, 100))  # Raises error.
# c_const[10] = 10  # Raises error.

c.value = 100
print(c_const.value)

print(c_const)
print(c_const.__dict__)
print(type(c_const.__dict__))
print(type_extract(c_const.__dict__))
# c_const.__dict__['value'] = 200

s = SubCheck()
s_const = to_const(s)

c.print_self()
c_const.print_self()
s_const.print_self()
# s_const.extra()  # Raises error.
# s_const.mutate()  # Raises error.
x = s_const.more()
print(x)
print(type(x))
# x[:] = []  # Raises error.

obj = to_const([1, 2, [10, 10]])
print(is_const(obj))
print(is_const([]))
# obj[1] = 3  # Raises error.
# to_mutable(obj)[1] = 10  # Raises error.
to_mutable(obj, force=True)[1] = 10
print(obj)

for i in obj:
    print(i)
    if isinstance(i, list):
        print(is_const(i))
        # i[0] = 10  # Raises error.
