#!/usr/bin/env python
from __future__ import print_function, absolute_import

# import unittest
# import pymodule
from pymodule.tpl import scalar_type as st
from pymodule.tpl.tpl_def import Template, is_tpl_cls, is_tpl_of


print("\n".join(sorted(st.__dict__.keys())))
print(st.Base__T_double__U_int.type_tup)
print(st.Base__T_int__U_double.type_tup)

base_types = {
    (long, float): st.Base__T_int__U_double,
    (float, long): st.Base__T_double__U_int,
    }

def BaseTpl(T=long, U=float):
    types = (T, U)
    return base_types[types]

# Default class.
Base = BaseTpl()

print(Base)
print(BaseTpl(long, float))
print(BaseTpl(float, long))

# Should only define these classes once.
def _Child(T=long, U=float):
    Base = BaseTpl(T, U)

    class Child(Base):
        def __init__(self, t, u):
            Base.__init__(self, t, u)

        def pure(self, t):
            print("py: pure")
            return U(t)

        def optional(self, t):
            print("py: optional")
            return U(2 * t)

        def do_to(self, Tc, Uc):
            # Scalar conversion.
            ChildTc = ChildTpl(Tc, Uc)
            return ChildTc(Tc(self.t()), Uc(self.u()))

    # Change name for clarity.
    Child.__name__ = "Child__T_{}__U_{}".format(T.__name__, U.__name__)
    return Child

child_types = {
    (long, float): _Child(long, float),
    (float, long): _Child(float, long),
    }

def ChildTpl(T=long, U=float):
    types = (T, U)
    return child_types[types]

# Default instantiation.
Child = ChildTpl()

print(Child)
print(ChildTpl(long, float))
print(ChildTpl(float, long))

print(Child == Child)
print(ChildTpl(long, float) == ChildTpl(float, long))

c = Child(2, 5.5)
print(type(c))
c.pure(1)
c.optional(2)
c.dispatch(3)

cc = c.do_to(float, long)
print(type(cc))
cc.pure(1.5)
cc.optional(1.5)
help(cc.dispatch)
cc.dispatch(1.5)

print("---")
st.call_method(c)
st.call_method(cc)
