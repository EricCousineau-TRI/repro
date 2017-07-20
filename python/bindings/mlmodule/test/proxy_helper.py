import numpy as np

def np_is_arithmetic(a):
    """
    Determine if array is arithmetic (int, float, etc.) or something else
    (e.g., symbolic).
    """
    # @ref https://stackoverflow.com/questions/22471644/how-do-i-check-if-a-numpy-dtype-is-integral
    dtype = a.dtype
    # TODO(eric.cousineau): Can return more complex things.
    if np.issubdtype(dtype, np.integer):
        return 1
    elif np.issubdtype(dtype, np.floating):
        return 2
    else:
        return 0
