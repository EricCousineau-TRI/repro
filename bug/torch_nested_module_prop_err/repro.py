#!/usr/bin/env python3
from torch import nn

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.good_attr = 1

class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.bad_attr = 1

class Top(nn.Module):
    def __init__(self, nested):
        super().__init__()
        self.nested = nested

    @property
    def proxy_property(self):
        # This works with A, but will fail with B. However, depending on
        # how this descriptor is called, a different error will be returned.
        return self.nested.good_attr

def get_err_str(stmt):
    try:
        stmt()
        exit(1)
    except Exception as e:
        return str(e)

def main():
    prop = Top.proxy_property
    print("Descriptor (good):", prop)
    print("nested=A (good):")
    print(" ", Top(A()).proxy_property)
    print("nested=B, getattr (unexpected error):")
    print(" ", get_err_str(lambda: Top(B()).proxy_property))
    print("nested=B, fget (expected error):")
    print(" ", get_err_str(lambda: prop.fget(Top(B()))))

assert __name__ == "__main__"
main()

"""
Descriptor (good): <property object at 0x7f67aeb15548>
nested=A (good):
  1
nested=B, getattr (unexpected error):
  'Top' object has no attribute 'proxy_property'
nested=B, fget (expected error):
  'B' object has no attribute 'good_attr'
"""
