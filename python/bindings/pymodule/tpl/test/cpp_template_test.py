#!/usr/bin/env python
from __future__ import print_function, absolute_import

from pymodule.tpl.cpp_template import Template, TemplateClass, is_class_instantiation, is_instantiation_of

from pymodule.tpl.test import _cpp_template_test as m

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

print("---")
print("Generics")
tpl = Template("generics_test")
tpl.add_instantiation((int,), 1)
tpl.add_instantiation((float,), 2)
assert tpl[int] == 1
assert tpl[float] == 2
try:
    tpl[str]
except RuntimeError as e:
    print(e)
tpl.add_instantiation((object,), 3)
assert tpl[object] == 3
assert tpl[str] == 3
