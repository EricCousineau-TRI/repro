import numpy as np
import numpy.typing as npt

from pydrake.autodiffutils import AutoDiffXd
from pydrake.symbolic import Expression

# https://numpy.org/devdocs/reference/typing.html
print(npt.NDArray)
print(npt.NDArray[np.float64])
print(npt.NDArray[object])
print(npt.NDArray[Expression])
print(npt.NDArray[AutoDiffXd])
