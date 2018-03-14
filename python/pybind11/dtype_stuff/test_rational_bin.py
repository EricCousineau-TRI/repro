import numpy as np
# from test_rational_min import rational
# from test_rational import rational
from numpy.core.test_rational import rational

print(rational)
print(rational(1))

print(np.zeros((2, 2), rational))

r = rational(1)
print(id(r))
t = r
print(id(t))
r += rational(10)
print(r)
