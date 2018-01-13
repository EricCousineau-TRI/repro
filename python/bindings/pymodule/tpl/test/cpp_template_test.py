#!/usr/bin/env python
from __future__ import print_function, absolute_import

from pymodule.tpl.cpp_template import Template, TemplateClass, is_class_instantiation, is_instantiation_of

from pymodule.tpl.test import _cpp_template_test as m

print("---")
def expect_throw(func):
    try:
        func()
        assert False
    except RuntimeError as e:
        print(e)

print("Generics")
tpl = Template("generics_test")
tpl.add_instantiation((int,), 1)
tpl.add_instantiation((float,), 2)
assert tpl[int] == 1
assert tpl[float] == 2
expect_throw(lambda: tpl[str])
tpl.add_instantiation((object,), 3)
assert tpl[object] == 3
assert tpl[str] == 3
_, param = tpl.get_instantiation((str,))
assert param == (object,)

tpl.add_instantiation((int, int), 4)
assert tpl[int, int] == 4
expect_throw(lambda: tpl[int, str])
tpl.add_instantiation((int, object), 5)
assert tpl[int, int] == 4
assert tpl[int, str] == 5

tpl.add_instantiation((object, int), 6)
assert tpl[int, int] == 4
assert tpl[int, str] == 5
assert tpl[str, int] == 6

tpl.add_instantiation((object, object), 7)
assert tpl[int, int] == 4
assert tpl[int, str] == 5
assert tpl[str, int] == 6
assert tpl[str, str] == 7

# Try ambiguous.
tpl.add_instantiation((object, object, object), 8)
tpl.add_instantiation((int, int, int), 9)
assert tpl[str, str, str] == 8
assert tpl[int, int, int] == 9
expect_throw(lambda: tpl.add_instantiation((object, object, int), 10))

assert tpl.get_param_list(9) == [(int, int, int)]

print("---")
print("Types")
print(m.template_type)
m.template_type[int]()
m.template_type[float]()
func = m.template_type[int]
assert is_instantiation_of(m.template_type[int], m.template_type)

m.template_list[int]()
m.template_list[int, float, m.SimpleType]()

print("---")
print("Class")
print(m.SimpleTemplate)
print(m.SimpleTemplateTpl)
print(m.SimpleTemplateTpl[int])
print(m.SimpleTemplateTpl[int, float, m.SimpleType])
assert is_class_instantiation(m.SimpleTemplate)
assert is_instantiation_of(m.SimpleTemplateTpl[int, float, m.SimpleType], m.SimpleTemplateTpl)

cls = m.SimpleTemplateTpl[int, float, m.SimpleType]
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
print(m.template_bool)
print(m.template_bool.param_list)
m.template_bool[False]()
m.template_bool[True]()
m.template_bool[0]()
m.template_bool[1]()

print(m.template_int)
print(m.template_int.param_list)
for i in [0, 1, 2, 5]:
    m.template_int[i]()
