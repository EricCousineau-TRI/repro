#!/usr/bin/env python
import weakref
# @ref https://docs.python.org/2/library/weakref.html
class Obj(list):
    pass

# Create instance, strong reference, and weak reference.
x = Obj([1, 2, 3])
x_strong = x
x_weak = weakref.ref(x)

# Delete original reference. Strong reference keeps alive.
print("del x")
del x
print("strong ref: {}".format(x_strong))
print("weak ref: {}".format(x_weak()))

# Delete strong reference. Weak reference does not keep alive.
print("\ndel x_strong")
del x_strong
print("weak ref: {}".format(x_weak()))
assert(x_weak() is None)
