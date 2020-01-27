#!/usr/bin/env python3

# Working pure-Python version of desired behavior in this issue, I think?
# https://github.com/pybind/pybind11/issues/1640

class Original:
    def stuff(self):
        return 1


class ExtendA(Original):
    def extra_a(self):
        return 2


class ExtendB(Original):
    def extra_b(self):
        return 3


def main():
    obj = Original()
    obj.__class__ = ExtendA
    assert obj.stuff() == 1
    assert isinstance(obj, Original)
    assert obj.extra_a() == 2
    assert type(obj) == ExtendA

    obj = Original()
    obj.__class__ = ExtendB
    assert obj.stuff() == 1
    assert isinstance(obj, Original)
    assert obj.extra_b() == 3
    assert type(obj) == ExtendB

    print("[ Done ]")


assert __name__ == "__main__"
main()
