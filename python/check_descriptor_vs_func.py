#!/usr/bin/env python3

import inspect


class Example:
    @staticmethod
    def my_static(): pass

    def my_method(self): pass

    @property
    def my_readonly(self): pass


def is_descriptor(x):
    return hasattr(x, "__get__")
    # The following also do not pass.
    # return hasattr(x, "__get__") and hasattr(x, "__set__")
    # return inspect.ismethoddescriptor(x)
    # return inspect.isdatadescriptor(x)
    # return inspect.isgetsetdescriptor(x)
    # return inspect.ismemberdescriptor(x)


def my_func(): pass


def main():
    assert is_descriptor(Example.my_static)
    assert is_descriptor(Example.my_method)
    assert is_descriptor(Example.my_readonly)

    assert not is_descriptor(my_func)  # FAIL: How to fix this?

    # TODO: Dunno what this should be.
    assert is_descriptor(Example().my_static)
    assert is_descriptor(Example().my_method)


assert __name__ == "__main__"
main()
