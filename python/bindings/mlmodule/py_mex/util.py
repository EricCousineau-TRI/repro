#!/usr/bin/env python

# Mechanism meant to store objects and refer to them with C-friendly
# types, that can then be passed to and unpacked by MATLAB.
class Erasure(object):
    def __init__(self):
        self._values = []
        self._occupied = []
    def store(self, value):
        i = -1
        for (i, occ) in enumerate(self._occupied):
            if not occ:
                break
        if i == -1:
            i = self._size()
            self._resize(self._size() + 4)
        assert(self._values[i] is None)
        assert(not self._occupied[i])
        self._values[i] = value
        self._occupied[i] = True
        return i
    def dereference(self, i, keep=False):
        assert(i < self._size())
        assert(self._occupied[i])
        value = self._values[i]
        if not keep:
            self._values[i] = None
            self._occupied[i] = False
        return value
    def _size(self):
        return len(self._values)
    def _resize(self, new_sz):        
        assert(new_sz >= self._size())
        dsz = new_sz - self._size()
        self._values += [None] * dsz
        self._occupied += [False] * dsz

if __name__ == "__main__":
    # Test erasure
    i1 = erasure.store(1)
    i2 = erasure.store({"hello": 1})
    assert(erasure.dereference(i1) == 1)
    assert(erasure.dereference(i2) == {"hello": 1})
