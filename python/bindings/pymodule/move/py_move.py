#!/usr/bin/env python
import sys

def _check_unique(obj):
    assert obj is not None
    ref_count = sys.getrefcount(obj)
    assert ref_count == 2, "Got ref count: {}".format(ref_count)

class PyMove(object):
    """ Provide a wrapper to permit passing an object to be owned by C++ """
    def __init__(self, obj):
        assert obj is not None
        self._obj = obj

    def release(self):
        obj = self._obj
        self._obj = None
        _check_unique(obj)
        return obj


def move(obj):
    return PyMove(obj)


if __name__ == '__main__':
    obj = [1, 2, 3]
    try:
        obj_mv = move(obj).release()
    except AssertionError, e:
        print("As expected")

    mv = move(obj)
    obj = None
    mv.release()
    print("Good")
