#!/usr/bin/env python
import sys
import weakref

def _check_unique(obj):
    assert obj is not None
    
class PyMove(object):
    """ Provide a wrapper to permit passing an object to be owned by C++ """
    def __init__(self, obj):
        assert obj is not None
        self._obj = obj

    def release(self):
        print("- release pre: {}".format(sys.getrefcount(self._obj)))
        obj = self._obj
        self._obj = None
        ref_count = sys.getrefcount(obj)
        print("- release post: {}".format(ref_count))
        # Cannot use `assert ...`, because it will leave a latent reference?
        # Consider a `with` reference?
        if ref_count != 2:
            obj = None
            raise AssertionError("Got ref count: {}".format(ref_count))
        else:
            return obj


def move(obj):
    return PyMove(obj)


if __name__ == '__main__':
    def main():
        obj = [1, 2, 3]
        print("- pre 1: {}".format(sys.getrefcount(obj)))
        mv = move(obj)
        print("- pre 2: {}".format(sys.getrefcount(obj)))
        try:
            # This increases the refcount?
            mv.release()
            pass
        except AssertionError:
            print("As expected")
        print("- post 1: {}".format(sys.getrefcount(obj)))
        # del obj_mv
        # del _
        # print(globals())
        print("- post 2: {}".format(sys.getrefcount(obj)))

        print("---")
        mv = move(obj)
        print("- pre: {}".format(sys.getrefcount(obj)))
        obj = None
        # obj = None
        mv.release()
        print("Good")

    main()
