#!/usr/bin/env python

import inject_module

try:
    print(inject_module.func())
    assert False
except Exception as e:
    assert e.message == "global name 'y' is not defined"
    print("Good: Expected error")

# Now try injecting.
inject_module.__dict__['y'] = 10
value = inject_module.func()
print(value)
assert value == 11
print("[ Done ]")
