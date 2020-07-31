from pymodule.tpl.cpp_types import type_name, type_canonical

class TestClass(object):
    pass

print(type_canonical(int))
print(type_name(float))
print(type_name(str))
print(type_canonical(TestClass))
print(type_name(TestClass))
