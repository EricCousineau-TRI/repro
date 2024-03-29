import functools

import numpy as np


def vectorize(**kwargs):
    # Decorator form of `np.vectorize`.
    return functools.partial(np.vectorize, **kwargs)


@vectorize(cache=True)
def op_binary(a, b):
    print(a, b)
    return a + b


@vectorize(cache=True)
def op_returns_tuple(a):
    return a, 2 * a


def main():
    a = [1, 2, 3]
    b = [10, 20, 30]
    c = op_binary(a, b)
    print(c)
    a2 = [4, 5, 6]
    b2 = [40, 50, 60]
    c2 = op_binary(a2, b2)
    print(c2)

    print()
    d = op_returns_tuple(a)
    print(d)



if __name__ == "__main__":
    main()


"""
Output:

1 10
2 20
3 30
[11 22 33]
4 40
5 50
6 60
[44 55 66]

(array([1, 2, 3]), array([2, 4, 6]))
"""
