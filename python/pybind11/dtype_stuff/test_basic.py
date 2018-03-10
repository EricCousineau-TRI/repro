import numpy as np

from __main__ import Custom

a = Custom(1)
print(a == a)

print(np.dtype(Custom))
print(Custom.dtype)
print(type(Custom.dtype))

class Stuff(object): pass

av = np.array([[Custom(1), Custom(2)], [Custom(3), Custom(4)]], dtype=Custom)
print(a)
print(av)
print(av == av)
print(av)
print(av.dtype)

av_bad = np.array([a, a])
print(av_bad == av_bad)
print(np.dtype(Stuff))
print(np.dtype(Custom))
