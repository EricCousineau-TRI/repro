#!/usr/bin/env python
from __future__ import print_function, absolute_import

from pymodule.tpl.cpp_template import TemplateClass, is_class_instantiation, is_instantiation_of

from pymodule.tpl.test import _scalar_type_test as m

# Default class.
BaseTpl = m.BaseTpl
Base = m.Base

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


def child_template_converter(ChildTpl, BaseTpl, param_list=BaseTpl.param_list):
    converter = BaseTpl.Converter()
    for to_param in param_list:
        for from_param in param_list:
            if to_param == from_param:
                continue
            cls_to = ChildTpl.get_instantiation(to_param)
            def converter_func(obj_from):
                return cls_to(obj_from)
            converter.Add(to_param, from_param, converter_func)
    return converter


# Should only define these classes once.
def _ChildTpl_instantiation(param):
    T, U = param
    Base = BaseTpl[T, U]

    class Child(Base):
        def __init__(self, *args, **kwargs):
            if args and is_instantiation_of(type(args[0]), ChildTpl):
                # Handle copy constructor overload.
                Base.__init__(self, *args, **kwargs)
            else:
                self._init(*args, **kwargs)

        def _init(self, t, u):
            Base.__init__(
                self, t, u, child_template_converter(ChildTpl, BaseTpl))

        def pure(self, value):
            print("py: pure [{}]".format(type(self).__name__))
            return U(2 * value)

        def optional(self, value):
            print("py: optional [{}]".format(type(self).__name__))
            return U(3 * value)

    return Child


ChildTpl = TemplateClass('ChildTpl')
ChildTpl.add_instantiations(_ChildTpl_instantiation, BaseTpl.param_list)
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
