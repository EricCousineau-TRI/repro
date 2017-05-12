def stuff_py(is_int=False):
    if is_int:
        return 1
    else:
        return 1.

def stuff_numpy(is_mat=False):
    import numpy as np
    import numpy.matlib as nm
    if is_mat:
        # Causes crash due to mkl.so
        return nm.matrix([1, 2, 3])
    else:
        return np.array([1, 2, 3])

def stuff_matlab():
    # Does not work
    import matlab
    return matlab.double(1.)

class Test:
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name
