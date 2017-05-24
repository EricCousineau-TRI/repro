def stuff_py(is_int=False):
    if is_int:
        return 1
    else:
        return 1.

def stuff_numpy(is_mat=False):
    import numpy as np
    import numpy.matlib as nm
    if is_mat:
        # Causes crash due to mkl.so if this is printed in MATLAB window
        return nm.matrix([1, 2, 3])
    else:
        return np.array([1, 2, 3])

def stuff_matlab():
    # Does not work
    import matlab
    return matlab.double(1.)

def passthrough(x):
    """ Hack to explicitly get MATLAB's conversion """
    return x

class Test:
    def __init__(self, name):
        import numpy as np
        self.name = name
        self.nparray = np.array([1, 2, 3])
    
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name

    def get_nparray(self):
        return self.nparray
    
    def set_nparray(self, nparray):
        self.nparray = nparray

    def do_stuff(self):
        print(type(self.name), self.name)
        print(type(self.nparray), self.nparray)
