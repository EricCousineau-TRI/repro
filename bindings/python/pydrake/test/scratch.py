def stuff_py(is_int=False):
    if is_int:
        return 1
    else:
        return 1.

def stuff_numpy():
    import numpy as np
    import numpy.matlib as nm
    return nm.matrix([1, 2, 3])

def stuff_matlab():
    # Does not work
    import matlab
    return matlab.double(1.)
