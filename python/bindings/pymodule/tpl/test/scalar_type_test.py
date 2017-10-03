#!/usr/bin/env python
from __future__ import print_function, absolute_import

# import unittest
from pymodule.tpl import scalar_type as st
from pymodule.tpl.tpl_def import Template, ChildTemplate, is_tpl_cls, is_tpl_of

BaseTpl = Template(
    name = 'Base',
    param_names = ('T', 'U'),
    param_defaults = (long, float))

BaseTpl.add_instantiation(
    (long, float), st.Base__T_int__U_double)
BaseTpl.add_instantiation(
    (float, long), st.Base__T_double__U_int)

# Default class.
Base = BaseTpl()

print(Base)
print(BaseTpl(long, float))
print(BaseTpl(float, long))
assert is_tpl_cls(Base)
assert is_tpl_of(Base, BaseTpl)

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
    return Child

ChildTpl = ChildTemplate(
    name = 'Child',
    parent = BaseTpl)

ChildTpl.add_instantiations_with_func(_Child)

# Default instantiation.
Child = ChildTpl()

print(Child)
print(ChildTpl(long, float))
print(ChildTpl(float, long))

# Check type identity persistence.
print(Child == ChildTpl())
print(ChildTpl(long, float) == ChildTpl(float, long))

assert is_tpl_cls(Child)
assert is_tpl_of(Child, ChildTpl)

# Check default instantiation.
assert issubclass(Child, Base)
# Check other instantiation.
assert issubclass(ChildTpl(float, long), BaseTpl(float, long))

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
