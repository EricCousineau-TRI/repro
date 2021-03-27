#!/usr/bin/env python3
import torch

class A(torch.nn.Module):
    @property
    def attr(self):
        return self.bad_prop

def get_err_str(stmt):
    try:
        stmt()
        exit(1)
    except Exception as e:
        return str(e)

def main():
    prop = A.attr
    print("Descriptor (good):", prop)
    print("A.attr (unexpected error):")
    print(" ", get_err_str(lambda: A().attr))
    print("prop.fget(A) (expected error):")
    print(" ", get_err_str(lambda: prop.fget(A())))

assert __name__ == "__main__"
main()
