#!/usr/bin/env python
from __future__ import print_function, absolute_import

# import unittest
# import pymodule
from pymodule.tpl import scalar_type as st

print("\n".join(sorted(st.__dict__.keys())))
print(st.Base__T_double__U_int.type_tup)
print(st.Base__T_int__U_double.type_tup)

base_types = {
    (long, float): st.Base__T_int__U_double,
    (float, long): st.Base__T_double__U_int,
    }

def BaseT(*args):
    return base_types[args]

# Default class.
Base = BaseT(long, float)

print(Base)
print(BaseT(long, float))
print(BaseT(float, long))

# Should only define these classes once.
def _Child(T, U):
    Base = BaseT(T, U)
    class Child(Base):
        def __init__(self, t, u):
            Base.__init__(self, t, u)

        def pure(self, u):
            print("py: pure")
            return T(u)

        def optional(self, u):
            print("py: optional")
            return T(2 * u)

    return Child

child_types = {
    (long, float): _Child(long, float),
    (float, long): _Child(float, long),
    }

def ChildT(*args):
    return child_types[args]
Child = ChildT(long, float)

print(Child)
print(ChildT(long, float))
print(ChildT(float, long))

print(Child == Child)
print(ChildT(long, float) == ChildT(float, long))
