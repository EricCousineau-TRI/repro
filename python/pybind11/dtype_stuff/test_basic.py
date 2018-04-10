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
    print(Custom)
    print(np.dtype(Custom))
    print(Custom.dtype)
    print(type(Custom.dtype))
    print(np.dtype(Unknown))
    print(np.dtype(Custom))

def check_id():
    a = Custom(1)
    print(id(a))
    print(id(a.self()))
    a += 1
    print(id(a))
    # b = a
    # print(id(b))
    # a.incr()
    # print(id(a))
    # a += 1
    # print(id(a))

def check_op_min():
    a = Custom(1)
    b = a
    a += 10
    print(a, b)
    av = np.array([a])
    print(a + a, av + av)
    print(-a, -av)
    print(a == a, av == av)
    print(a * a, av * av)

def check_op():
    a = Custom(1)
    print(a == a)
    av = np.array([a])  # Implicit cast.
    dv = av.astype(float)
    print(dv)
    ov = av.astype(object)
    print(av == av)  # non-logical
    print(ov == ov)  # logical

    print("---")
    a2v_bad = np.array([a, 10])  # Cannot mix easily.
    print(repr(a2v_bad))
    print(">>>")
    print(repr(a2v_bad.astype(Custom)))  # But we can cast.

    av = np.array([[Custom(1), Custom(2)], [Custom(3), Custom(4)]], dtype=Custom)
    print(av)
    print(av == av)
    print av * av
    print av < av
    print -av
    print np.power(av, av)  # ... How does this even work???
    xv = av.astype(float)
    print(repr(xv))
    print(repr(xv.astype(Custom)))
    ov = av.astype(object)
    print(repr(ov))
    print(repr(ov.astype(Custom)))

    a0 = np.zeros((2, 2), Custom)
    print(a0)
    try:
        av[0] = "aow"
        print("dflkj")
    except:
        print("error")

def check_rational():
    x = np.zeros((2, 2), rational)
    print(repr(x))
    y = np.array([rational()])
    print(repr(y))

def check_dtor():
    x = Custom(1)
    print(x)

def check_mutate():
    av = np.array([Custom(1), Custom(2)])
    mutate(av)
    print(av)
    print(av == av)

def check_func():
    def func():
        x = np.array([Custom(1), Custom(2)])
        print("called: ", x)
        x += Custom(10)
        return x
    call_func(func)

# check_meta()

# check_rational()
# check_bad()
# check_zero()
# check_op()
# check_id()
# check_op_min()

# check_dtor()

# check_mutate()
sys.stdout = sys.stderr

import subprocess
subprocess.Popen(
    "export -p | sed 's# PWD=# OLD_PWD=#g' > /tmp/env.sh",
    shell=True)

check_func()

# import code
# code.InteractiveConsole(locals=globals()).interact()
