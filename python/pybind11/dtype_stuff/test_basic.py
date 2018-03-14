import numpy as np
import os
import sys

class Unknown(object): pass

def check_zero():
    a0 = np.zeros((2, 2), Custom)
    print(a0)

def check_meta():
    print(Custom)
    a = Custom(1)
    print(str(a), repr(a))
    av = np.array([a])  # Implicit cast.
    print("---")
    a2v_bad = np.array([a, 10])  # Cannot mix easily.
    print(repr(a2v_bad))
    a2v = np.array([a, 10], dtype=Custom)  # Cannot mix easily.
    print(repr(a2v))
    ov = av.astype(object)
    print(av == av)
    print(ov == ov)
    print(Custom)
    print(np.dtype(Custom))
    print(Custom.dtype)
    print(type(Custom.dtype))
    print(np.dtype(Unknown))
    print(np.dtype(Custom))

def check_op():
    a = Custom(1)
    print(a == a)
    av = np.array([[Custom(1), Custom(2)], [Custom(3), Custom(4)]], dtype=Custom)
    print(av)
    print(av == av)
    print av * av
    print av < av
    print -av
    print np.power(av, av)
    xv = av.astype(float)
    print(repr(xv))
    print(repr(xv.astype(Custom)))
    ov = av.astype(object)
    print(repr(ov))
    print(repr(ov.astype(Custom)))

    a0 = np.zeros(2, 2, Custom)
    print(a0)

def check_rational():
    x = np.zeros((2, 2), rational)
    print(repr(x))
    y = np.array([rational()])
    print(repr(y))

# check_meta()

# check_rational()
# check_bad()
check_zero()
# # check_op()
