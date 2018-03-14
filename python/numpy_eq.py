# Purpose: See if there's a way to not have == return a logical.
import numpy as np

class Custom(object):
    def __init__(self, value):
        self.value = value

    def __eq__(self, rhs):
        return "eq({}, {})".format(self, rhs)

    def __str__(self):
        return repr(self.value)

    def __repr__(self):
        return repr(self.value)


def main():
    # Scalar case works.
    a = Custom('a')
    print(a == a)
    # Not what we want.
    av = np.array([a, a])
    print(np.equal(av, av))
    # Try custom ufunc:
    generic_equal = np.frompyfunc(lambda a, b: a == b, 2, 1)
    print(generic_equal(av, av))
    # Try replacing.
    np.set_numeric_ops(equal=generic_equal)
    print(av == av)
    print(np.equal(av, av))
    # Now replace original ufunc.
    np.equal = generic_equal
    print(np.equal(av, av))

def exec_lines(func):
    import inspect
    import sys

    lines, _ = inspect.getsourcelines(func)
    cur = {}
    for line in lines[1:]:
        line_trim = line[4:].rstrip()
        if not line_trim:
            continue
        print(">>> {}".format(line_trim))
        exec line_trim in globals(), cur

exec_lines(main)
