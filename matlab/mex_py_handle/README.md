An attempt to pass MATLAB function handles to Python, possibly leveraging MATLAB's Python bridge.

# TODO

* Check `pybind11` virtual inheritance running under MATLAB
    * See if MATLAB gets angry about releasing GIL for virtual calls.  
        * Presently, seems to be failing.
        * TODO: Do an `py::eval` from `pybind11`, to see if it can call MATLAB without problem.
* Try passing in a generic MATLAB object to Python.
    * Consider sticking to Python-inherited MATLAB objects only?
    * Pass other classes as opaque references?
* Test out what a simple Python-bound virtual inheritence structure may look like.
* Check matrix inversion with `drake::symbolic::Expression`.
* See if there is a way to separate LD loading of boost, MKL, etc., so that NumPy + MATLAB can get along.
    * Example: http://adared.ch/matpy/
    * Choosing SciPy libs:
        * https://www.scipy.org/scipylib/building/linux.html
        * Disable MKL altogether for Python???
* Test on Mac.
* If useful, merge functionality into `PyProxy`

# Done

* Passing C function pointers between MEX and Python
* Calling MATLAB functions from Python
    * (Ensuring that GIL stuff doesn't get screwy)
* Wrapping MATLAB function references
* Reference counting MATLAB function references
    * (So that they can be stored)

# References

* Possibly use `ctypes` for erasure from MATLAB back to Python, like:
    * https://github.com/dgorissen/pymatopt
* https://stackoverflow.com/questions/15011674/is-it-possible-to-dereference-variable-ids
* https://stackoverflow.com/questions/3245859/back-casting-a-ctypes-py-object-in-a-callback
* Concerns:
    * See comments in `mex_py_proxy.cpp` for more details.
