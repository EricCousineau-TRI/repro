#!/usr/bin/env python
from __future__ import print_function, absolute_import

# import unittest
from pymodule.tpl import scalar_type as st
from pymodule.tpl.py_tpl import Template, ChildTemplate, is_tpl_cls, is_tpl_of

# # Default class.
BaseTpl = st.BaseTpl
Base = st.Base


print(Base)
print(BaseTpl[int, float])
print(BaseTpl[float, int])
assert is_tpl_cls(Base)
assert is_tpl_of(Base, BaseTpl)

# Test direct inheritance.
class ChildDirect(Base):
    def __init__(self, t, u):
        Base.__init__(self, t, u)
    def pure(self, t):
        print("py direct: pure")
        return 1.
    def optional(self, t):
        print("py direct: optional")
        return 2.

# Should only define these classes once.
def _ChildTpl[T, U]:
    Base = BaseTpl[T, U]
    class Child(Base):
        def __init__(self, t, u, other=None):
            # Add the same converter per instance.
            if other is not None:
                Base.__init__(self, other, _Child_converter())
            else:
                Base.__init__(self, t, u, _Child_converter())

        def pure(self, t):
            print("py: pure [{}]".format(type(self).__name__))
            return U(t)

        def optional(self, t):
            print("py: optional [{}]".format(type(self).__name__))
            return U(2 * t)

        def do_to(self, Tc, Uc):
            # Scalar conversion.
            ChildTc = ChildTpl[Tc, Uc]
            # out = ChildTc(Tc(self.t()), Uc(self.u()))
            out = ChildTc(self)
            print("py.do_to:")
            out.dispatch(Tc())
            print("  {} - {}".format(out.t(), out.u()))
            return out
    return Child


ChildTpl = ChildTemplate(
    name = 'Child',
    parent = BaseTpl)
ChildTpl.add_instantiation_factory(_ChildTpl)


def _Child_converter():
    converter = st.BaseConverter()
    def add_conversion(param_to, param_from):
        cls_from = ChildTpl[*param_from]
        cls_to = ChildTpl[*param_to]
        def func(obj_from):
            print("py.1: Sanity check")
            assert isinstance(obj_from, cls_from)
            print("py.2: Call method")
            obj_to = obj_from.do_to(*param_to)
            assert isinstance(obj_to, cls_to)
            print("py.3: Return")
            return obj_to
        converter.Add(param_to, param_from, func)
    add_conversion((int, float), (float, int))
    add_conversion((float, int), (int, float))
    return converter


# Default instantiation.
Child = ChildTpl[[]]

print(Child)
print(ChildTpl[int, float])
print(ChildTpl[float, int])

# Check type identity persistence.
print(Child == ChildTpl[[]])
print(ChildTpl[int, float] == ChildTpl[float, int])

assert is_tpl_cls(Child)
assert is_tpl_of(Child, ChildTpl)

# Check default instantiation.
assert issubclass(Child, Base)
# Check other instantiation.
assert issubclass(ChildTpl[float, int], BaseTpl[float, int])

cd = ChildDirect(2, 5.5)
print(type(cd))
cd.pure(1)
cd.optional(2)
cd.dispatch(3)
print("---")

c = Child(2, 5.5)
print(type(c))
c.pure(1)
c.optional(2)
c.dispatch(3)
print("---")

cc = c.do_to(float, int)
print(type(cc))
cc.pure(1.5)
cc.optional(1.5)
cc.dispatch(1.5)

print("---")
st.call_method(cd)
st.call_method(c)
st.call_method(cc)

print("---")
func = lambda: ChildTpl[float, int](6.5, 3)
print("Check")
owne = st.take_ownership(func)
owne.dispatch(3.5)
print("Good")

print("---")
cc_c = st.do_convert(c)
print("Try dispatch")
cc_c.dispatch(2.5)
print("Good to go")
