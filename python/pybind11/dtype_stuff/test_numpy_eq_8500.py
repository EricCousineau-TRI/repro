 # Purpose: Check why NumPy does weird things...
import numpy as np
import warnings

warnings.filterwarnings("default", category=DeprecationWarning, message="numpy equal will not")
warnings.filterwarnings("default", category=DeprecationWarning, message="elementwise == comparison failed")


class NonConvertible(object):
    def __nonzero__(self):
        m.trigger()
        raise ValueError("do not call")


class Custom(object):
    def __eq__(self, other):
        return NonConvertible()

x = Custom()
y = Custom()
xv = np.array([x, y])
yv = np.array([x, x])

print(xv == xv)
print(xv == yv)

try:
    bool(x == x)
    exit(1)
except ValueError:
    pass

print("Hello")
