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

def BaseT(T=long, U=float):
    types = (T, U)
    return base_types[types]

# Default class.
Base = BaseT()

print(Base)
print(BaseT(long, float))
print(BaseT(float, long))

# Should only define these classes once.
def _Child(T=long, U=float):
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

    # Change name for clarity.
    Child.__name__ = "Child__T_{}__U_{}".format(T.__name__, U.__name__)
    return Child

child_types = {
    (long, float): _Child(long, float),
    (float, long): _Child(float, long),
    }

def ChildT(T=long, U=float):
    types = (T, U)
    return child_types[types]

# Default instantiation.
Child = ChildT()

print(Child)
print(ChildT(long, float))
print(ChildT(float, long))

print(Child == Child)
print(ChildT(long, float) == ChildT(float, long))

c = Child(2, 5.5)
c.pure(1)
c.optional(2)
c.dispatch(3)
