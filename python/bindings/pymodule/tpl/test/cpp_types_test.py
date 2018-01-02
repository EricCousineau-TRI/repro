from pymodule.tpl.cpp_types import type_registry

tr = type_registry

class TestClass(object):
    pass

print(tr.GetPyTypeCanonical(int))
print(tr.GetName(float))
print(tr.GetName(str))
print(tr.GetPyTypeCanonical(TestClass))
print(tr.GetName(TestClass))
