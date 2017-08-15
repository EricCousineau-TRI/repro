#!/usr/bin/env python
import weakref

class Obj(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)

def func():
    x = Obj([1, 2, 3])
    y = weakref.ref(x)
    def a():
        print("strong ref: {}".format(x))
    def b():
        xr = y()
        if xr is not None:
            print("weakref: {}".format(xr))
        else:
            print("weakref expired")
    a()
    b()
    return (a, b)

(a, b) = func()
a()
b()

del a
b()
