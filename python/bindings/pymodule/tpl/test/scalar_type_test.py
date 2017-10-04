#!/usr/bin/env python
from __future__ import print_function, absolute_import

# import unittest
from pymodule.tpl import scalar_type as st
from pymodule.tpl.py_tpl import Template, ChildTemplate, is_tpl_cls, is_tpl_of

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
def _def_Child(T=long, U=float):
    Base = BaseTpl(T, U)
    class Child(Base):
        def __init__(self, t, u):
            # Add the same converter per instance.
            Base.__init__(self, t, u, _convert_Child())

        def pure(self, t):
            print("py: pure [{}]".format(type(self).__name__))
            return U(t)

        def optional(self, t):
            print("py: optional [{}]".format(type(self).__name__))
            return U(2 * t)

        def do_to(self, Tc, Uc):
            # Scalar conversion.
            ChildTc = ChildTpl(Tc, Uc)
            out = ChildTc(Tc(self.t()), Uc(self.u()))
            print("py.do_to:")
            out.dispatch(Tc())
            print("  {} - {}".format(out.t(), out.u()))
            return out
    return Child

def _convert_Child():
    converter = st.BaseConverter()
    def add_conversion(params_to, params_from):
        cls_from = ChildTpl(*params_from)
        cls_to = ChildTpl(*params_to)
        def func(obj_from):
            print("py.1: Sanity check")
            assert isinstance(obj_from, cls_from)
            print("py.2: Call method")
            obj_to = obj_from.do_to(*params_to)
            assert isinstance(obj_to, cls_to)
            print("py.3: Return")
            return obj_to
        converter.Add(params_to, params_from, func)
    add_conversion((long, float), (float, long))
    add_conversion((float, long), (long, float))
    return converter

ChildTpl = ChildTemplate(
    name = 'Child',
    parent = BaseTpl)

ChildTpl.add_instantiations_with_func(_def_Child)

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

cc = c.do_to(float, long)
print(type(cc))
cc.pure(1.5)
cc.optional(1.5)
cc.dispatch(1.5)

print("---")
st.call_method(cd)
st.call_method(c)
st.call_method(cc)

print("---")
cc_c = st.do_convert(c)
print("Try dispatch")
cc_c.dispatch(2.5)
print("Try again")
