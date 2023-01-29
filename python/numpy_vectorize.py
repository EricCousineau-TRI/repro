import numpy as np


def op_binary(a, b):
    print(a, b)
    return a + b


def main():
    op = np.vectorize(op_binary, cache=True)
    a = [1, 2, 3]
    b = [10, 20, 30]
    c = op(a, b)
    print(c)
    a2 = [4, 5, 6]
    b2 = [40, 50, 60]
    c2 = op(a2, b2)
    print(c2)


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
"""
