#!/usr/bin/env python
import sys
import weakref

def _check_unique(obj):
    assert obj is not None

# Can a `weakref` be used to get `refcount`?
    
class PyMove(object):
    """ Provide a wrapper to permit passing an object to be owned by C++ """
    def __init__(self, obj):
        assert obj is not None
        self._obj = obj

    def release(self):
        obj = self._obj
        self._obj = None
        ref_count = sys.getrefcount(obj)
        # Cannot use `assert ...`, because it will leave a latent reference?
        # Consider a `with` reference?
        if ref_count > 2:
            obj = None
            raise AssertionError("Object refernce is not unique, got {} extra references".format(ref_count - 2))
        else:
            assert ref_count == 2
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
        except AssertionError, e:
            print("Got expected error: {}".format(e))
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
