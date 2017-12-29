from pymodule.tpl.cpp_tpl_types import type_registry

tr = type_registry

class TestClass(object):
    pass

print(tr.GetPyTypeCanonical(int))
print(tr.GetCppName(float))
print(tr.GetCppName(str))
print(tr.GetPyTypeCanonical(TestClass))
print(tr.GetCppName(TestClass))
