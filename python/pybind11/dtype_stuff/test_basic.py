import numpy as np

from __main__ import Custom

a = Custom(1)

def check_bad():
    av_bad = np.array([Custom(1)])
    print(av_bad == av_bad)

def check_dtype():
    print(np.dtype(Custom))
    print(Custom.dtype)
    print(type(Custom.dtype))

    class Stuff(object): pass
    print(np.dtype(Stuff))
    print(np.dtype(Custom))

def check_op():
    print(a == a)
    av = np.array([[Custom(1), Custom(2)], [Custom(3), Custom(4)]], dtype=Custom)
    print(av)
    print(av == av)
    print av * av
    print av < av

check_op()
