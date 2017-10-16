#!/usr/bin/env python

class A:
    def __init__(self):
        print("A.__init__")
    def __del__(self):
        print("A.__del__")

class B(A):
    def __init__(self):
        A.__init__(self)
        print("B.__init__")
    def __del__(self):
        # Must explicitly call parent dtor.
        print("B.__del__")
        A.__del__(self)

obj = B()
def override():
    print("Override")
obj.__del__ = override
del obj

# Test with slots.
class C:
    __slots__ = '__del__'
    def __del__(self):
        print("Dtor")

obj = C()
obj.__del__ = override
del obj
