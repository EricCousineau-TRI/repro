# import subprocess
# subprocess.Popen(
#     "export -p | sed 's# PWD=# OLD_PWD=#g' | tee /tmp/env.sh",
#     shell=True)

import numpy as np
# from test_rational_min import rational
import test_rational_min
# from test_rational import rational
from numpy.core.test_rational import rational

# print(np.ones((2, 2), dtype=rational))
x = np.array([0.])
x[0] = rational(1)
print(x)
print(x + rational(1))
print(rational(1) + x)
# print(rational)
# print(rational(1))

# print(np.zeros((2, 2), rational))
#
# r = rational(1)
# print(id(r))
# t = r
# print(id(t))
# r += rational(10)
# print(r)
