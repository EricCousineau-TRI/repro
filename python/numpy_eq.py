# Purpose: See if there's a way to not have == return a logical.
import inspect
import numpy as np
import sys

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


def main():
    eq = lambda a, b: a == b
    generic_equal = np.frompyfunc(eq, 2, 1)

    a = Custom('a')
    b = Custom('b')
    print(a == b)

    av = np.array([a, b])
    bv = np.array([b, a])

    print(generic_equal(av, bv))
    print(np.equal(av, bv))

    np.set_numeric_ops(equal=generic_equal)
    print(np.equal(av, bv))
    print(av == bv)
    np.equal = generic_equal
    print(np.equal(av, bv))

def exec_lines(func):
    lines, _ = inspect.getsourcelines(func)
    cur = {}
    for line in lines[1:]:
        line_trim = line[4:]
        print(">> {}".format(line_trim.rstrip()))
        exec line_trim in globals(), cur

exec_lines(main)
