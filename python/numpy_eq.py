# Purpose: See if there's a way to not have == return a logical.

import numpy as np

class Custom(object):
    def __init__(self, value):
        self.value = value

    def __eq__(self, rhs):
        return "eq({}, {})".format(self, rhs)

    def __lt__(self, rhs):
        return "lhs({}, {})".format(self, rhs)

    def __str__(self):
        return repr(self.value)

    def __repr__(self):
        return repr(self.value)

equal = np.frompyfunc(Custom.__eq__, 2, 1)

a = Custom('a')
b = Custom('b')
print(a == b)

av = np.array([a, b])
bv = np.array([b, a])

print(equal(av, bv))
# print(np.equal(av, bv, dtype=Custom))
print(np.equal(av, bv))
