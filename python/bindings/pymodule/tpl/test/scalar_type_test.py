#!/usr/bin/env python
from __future__ import print_function, absolute_import

from pymodule.tpl.cpp_template import TemplateClass, is_class_instantiation, is_instantiation_of

from pymodule.tpl.test import _scalar_type_test as m

import sys
sys.stdout = sys.stderr

# Default class.
BaseTpl = m.BaseTpl
Base = m.Base

def child_template_converter(ChildTpl, BaseTpl, param_list=None):
    if param_list is None:
        param_list = BaseTpl.param_list
    converter = BaseTpl.Converter()
    for to_param in param_list:
        for from_param in param_list:
            if to_param == from_param:
                continue
            cls_to, _ = ChildTpl.get_instantiation(to_param)
            def converter_func(obj_from):
                return cls_to(obj_from)
            converter.Add(to_param, from_param, converter_func)
    return converter

def scalar_converter_init(ChildTpl, BaseTpl):
    def decorator(init_object):

        def init_wrapped(self, *args, **kwargs):
            if len(args) == 1 and is_instantiation_of(type(args[0]), ChildTpl):
                init_object.init_copy(self, args[0])
            else:
                scalar_converter = child_template_converter(ChildTpl, BaseTpl)
                init_object.init_normal(self, scalar_converter, *args, **kwargs)

        return init_wrapped

    return decorator

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


# Should only define these classes once.
@TemplateClass.define('ChildTpl', param_list=BaseTpl.param_list)
def ChildTpl(param, ChildTpl):
    T, U = param
    Base = BaseTpl[T, U]

    class Child(Base):
        @scalar_converter_init(ChildTpl, BaseTpl)
        class __init__(object):
            @staticmethod
            def init_copy(self, other):
                Base.__init__(self, other)
                self.extra = 'copied'

            @staticmethod
            def init_normal(self, scalar_converter, t, u):
                Base.__init__(self, scalar_converter, t, u)
                self.extra = 'normal'

        def pure(self, value):
            print("py extra: ", self.extra)
            print("py: pure [{}]".format(type(self).__name__))
            return U(2 * value)

        def optional(self, value):
            print("py extra: ", self.extra)
            print("py: optional [{}]".format(type(self).__name__))
            return U(3 * value)

    return Child

# Default instantiation.
Child, _ = ChildTpl.get_instantiation()

print(Child)
print(ChildTpl[int, float])
print(ChildTpl[float, int])

# Check type identity persistence.
assert Child == ChildTpl.get_instantiation()[0]
assert ChildTpl[int, float] == ChildTpl[int, float]
assert ChildTpl[int, float] != ChildTpl[float, int]

# Check default instantiation.
print("--------")
print(Child)
print(Base)
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
m.call_method(cd)
m.call_method(c)
m.call_method(cc)

print("---")
def factory():
    out = ChildTpl[float, int](6.5, 3)
    print(out)
    return out
print("Check")
owne = m.take_ownership(factory)
print("dispatch")
owne.dispatch(3.5)
print("Good")

print("---")
c.dispatch(3)
cc_c = m.do_convert(c)
print("Try dispatch")
cc_c.dispatch(2.5)
print("Good to go")
