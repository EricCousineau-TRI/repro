"""
Additional primitives for the Drake systems framework.
"""

import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.systems.primitives import (
    ConstantValueSource,
    ConstantVectorSource_,
)


def _is_array(x):
    # Determines if x could be ndarray-compatible.
    return isinstance(x, (np.ndarray, list))


def ConstantSource(value):
    """Sugar function to create either a ConstantVectorSource or
    ConstantValueSource based on an input value."""
    if _is_array(value):
        # Use template to permit different dtypes.
        return ConstantVectorSource_(value)
    else:
        return ConstantValueSource(AbstractValue.Make(value))
