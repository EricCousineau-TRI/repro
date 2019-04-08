# Alternative impl, motivated by: https://bitbucket.org/ericvsmith/namedlist
from functools import partial

import numpy as np


def _item_property(cls, index):
    return property(
        fget=lambda self: cls.__getitem__(self, index),
        fset=lambda self, value: cls.__setitem__(self, index, value))


def namedview(name, fields):

    class NamedView(object):
        def __init__(self, obj):
            assert len(obj) == len(fields)
            self._obj = obj

        def __getitem__(self, index):
            return self._obj.__getitem__(index)

        def __setitem__(self, index, value):
            self._obj.__setitem__(index, value)

        def __len__(self):
            return self._obj.__len__

        def __iter__(self):
            return self._obj.__iter__()

        def __repr__(self):
            return self._obj.__repr__()

        def __str__(self):
            return self._obj.__str__()

    NamedView.__name__ = NamedView.__qualname__ = name
    for i, field in enumerate(fields):
        setattr(NamedView, field, _item_property(NamedView, i))
    return NamedView


def test_main():
    MyView = namedview("MyView", ['a', 'b', 'c'])
    print(MyView)

    print("[ Simple List ]")
    value = [1, 2, 3]
    view = MyView(value)
    print(view.a)
    view[0] = 10
    print(view.a)
    view[1] = -100
    view.c = 1000
    print(view)
    print(value)
    view[:] = [111, 222, 333]
    print(view)
    print(value)

    print("[ Array 1D ]")
    array = np.array([4, 5, 6])
    aview = MyView(array)
    print(aview.a)
    aview[[1, 2]] = [50, 60]
    print(array)
    print(aview)

    # Maybe not useful, but meh.
    print("[ Array 2D ]")
    mat = np.eye(3)
    mview = MyView(mat)
    print(mview.a)
    print(mview[0, 0])
    mview.a[0] = 10
    print(mview.b)
    print(mat)


"""
Example output:

<class '__main__.MyView'>
[ Simple List ]
1
10
[10, -100, 1000]
[10, -100, 1000]
[111, 222, 333]
[111, 222, 333]
[ Array 1D ]
4
[ 4 50 60]
[ 4 50 60]
[ Array 2D ]
[ 1.  0.  0.]
1.0
[ 0.  1.  0.]
[[ 10.   0.   0.]
 [  0.   1.   0.]
 [  0.   0.   1.]]
"""



if __name__ == "__main__":
    test_main()
