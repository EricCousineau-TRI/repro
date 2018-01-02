#!/usr/bin/env python
from __future__ import print_function, absolute_import

# import unittest
from pymodule.tpl import scalar_type as st
from pymodule.tpl.cpp_template import TemplateClass, is_instantiation, is_instantiation_of

import sys
sys.stderr = sys.stdout

print("Types")
print(st.template_type)
st.template_type[int]()
st.template_type[float]()
func = st.template_type[int]
assert is_instantiation_of(st.template_type[int], st.template_type)

st.template_list[int]()
st.template_list[int, float, st.SimpleType]()

print("---")
print("Class")
print(st.SimpleTemplate)
print(st.SimpleTemplate[int])
print(st.SimpleTemplate[int, float, st.SimpleType])
assert is_instantiation(st.SimpleTemplate[int])
assert is_instantiation_of(st.SimpleTemplate[int, float, st.SimpleType], st.SimpleTemplate)

cls = st.SimpleTemplate[int, float, st.SimpleType]
s = cls()
print(s.size())
s.check[float]()

# Unbound
print(cls.check)
print(cls.check[float])
# Bound
print(s.check)
print(s.check[float])

print("---")
print("Literals")
print(st.template_bool)
print(st.template_bool.param_list)
st.template_bool[False]()
st.template_bool[True]()
st.template_bool[0]()
st.template_bool[1]()

print(st.template_int)
print(st.template_int.param_list)
for i in [0, 1, 2, 5]:
    st.template_int[i]()

# # Default class.
BaseTpl = st.BaseTpl
Base = st.Base

print("---")
# Test direct inheritance.
class ChildDirect(Base):
    def __init__(self, t, u):
        Base.__init__(self, t, u)
    def pure(self, t):
        print("py direct: pure [{}]".format(type(self).__name__))
        return 1.
    def optional(self, t):
        print("py direct: optional [{}]".format(type(self).__name__))
        return 2.


def child_template_converter(ChildTpl):
    BaseTpl = ChildTpl.parent
    converter = BaseTpl.Converter()
    for to_param in BaseTpl.param_list:
        for from_param in BaseTpl.param_list:
            if to_param == from_param:
                continue
            cls_to = ChildTpl.get_instantiation(to_param)
            def converter_func(obj_from):
                return cls_to(copy_from=obj_from)
            converter.Add(to_param, from_param, converter_func)
    return converter


# Should only define these classes once.
def _ChildTpl_instantiation(param):
    T, U = param
    Base = BaseTpl[T, U]

    class Child(Base):
        def __init__(self, *args, **kwargs):
            # Handle copy constructor overload:
            if "copy_from" in kwargs:
                copy_from = kwargs["copy_from"]
                Base.__init__(self, copy_from)
            else:
                self._init(*args, **kwargs)

        def _init(self, t, u):
            Base.__init__(self, t, u, child_template_converter(ChildTpl))

        def pure(self, value):
            print("py: pure [{}]".format(type(self).__name__))
            return U(2 * value)

        def optional(self, value):
            print("py: optional [{}]".format(type(self).__name__))
            return U(3 * value)

    return Child


ChildTpl = TemplateClass(
    name = 'ChildTpl',
    parent = BaseTpl)
ChildTpl.add_instantiations(_ChildTpl_instantiation)
# Default instantiation.
Child = ChildTpl.get_instantiation()

print(Child)
print(ChildTpl[int, float])
print(ChildTpl[float, int])

# Check type identity persistence.
assert Child == ChildTpl.get_instantiation()
assert ChildTpl[int, float] == ChildTpl[int, float]
assert ChildTpl[int, float] != ChildTpl[float, int]

assert is_instantiation(Child)
assert is_instantiation_of(Child, ChildTpl)

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
cc = c.DoTo[float, int]()
print(type(cc))
cc.pure(1.5)
cc.optional(1.5)
cc.dispatch(1.5)

print("---")
st.call_method(cd)
st.call_method(c)
st.call_method(cc)

print("---")
def factory():
    out = ChildTpl[float, int](6.5, 3)
    print(out)
    return out
print("Check")
owne = st.take_ownership(factory)
print("dispatch")
owne.dispatch(3.5)
print("Good")

print("---")
c.dispatch(3)
cc_c = st.do_convert(c)
print("Try dispatch")
cc_c.dispatch(2.5)
print("Good to go")
