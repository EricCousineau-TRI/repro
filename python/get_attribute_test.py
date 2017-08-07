#!/usr/bin/env python

class GetAttr(object):
    def __init__(self):
        self.value = 1
        pass
    def __getattr__(self, key):
        print("getattr: {}".format(key))
        return self.__dict__[key]

class GetAttribute(object):
    def __init__(self):
        self.value = 1
        pass
    def __getattribute__(self, key):
        print("getattribute: {}".format(key))
        return object.__getattribute__(self, key)
    def __setattr__(self, key, value):
        print("setattr: {}".format(key))
        return object.__setattr__(self, key, value)

a = GetAttr()
a.value = 10
print(a.value)
print("---")
b = GetAttribute()
b.value = 10
print(b.value)
