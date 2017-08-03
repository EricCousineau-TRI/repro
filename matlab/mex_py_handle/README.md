An attempt to pass MATLAB function handles to Python, possibly leveraging MATLAB's Python bridge.

Possibly use `ctypes` for erasure from MATLAB back to Python, like:
https://github.com/dgorissen/pymatopt

# Notes

    To convert MATLAB to Python:

        >> ML
        >> py.pass_through({arg1, arg2, ...})
        py.tuple(...)
        >> py_ml_sin = py.ml_wrap.wrap(@sin)
        >> py.call(py_ml_sin, 0)
        0

    To convert Python to MATLAB

        >> ML
        >> mex_args = mex_opaque({arg1, arg2, ...})


Example:

    Direct

        Python
            def call_func(f, arg1, arg2):
                f(arg1, arg2)

        MATLAB
            function my_func(arg1, arg2)
                disp(arg1)
                disp(arg2)
            end

            py_call_func_direct(@my_func, struct('a', 1), eye(3))

        Breakdown:
            py_call_func_direct
                Direct call.
                In MATLAB
                    Will intercept all args, wrap into opaque argument set ("rhs")
                    Pass to Python with opaque args
                In Python
                    Will invoke ctypes to pass func handle and arguments directly through

    Indirect
        # https://stackoverflow.com/questions/15011674/is-it-possible-to-dereference-variable-ids
        # https://stackoverflow.com/questions/3245859/back-casting-a-ctypes-py-object-in-a-callback
        # Concerns: https://stackoverflow.com/questions/18660433/matlab-mex-file-with-mexcallmatlab-is-almost-300-times-slower-than-the-correspon

        Python
            def call_func(f, x, str):
                call_matlab(f, 1, 2 * x, str + " world")

            from ctypes import *
            mx_array_t = c_void_p  # mxArray*
            mx_array_t_p = c_void_p  # mxArray**

            # int (mxArray* func_handle, void_p) # int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[])
            c_call_matlab_t = CFUNCTYPE(py_object, mx_array_t, c_int, mx_array_t_p, c_int, mx_array_t_p)

            # void* (void*) - but use ctypes to extract void* from py_object
            c_raw_to_py_t = ctypes.CFUNCTYPE(py_object, c_void_p)  # Use py_object so ctypes can cast to void*
            c_py_to_raw_t = ctypes.CFUNCTYPE(c_void_p, py_object)

            # py - Python
            # py_raw - Raw Python points (py_object, c_void_p)
            # ml, mx - MATLAB
            # mx_raw - void* representing mxArray* (mx_array_t, c_void_p)
            # mex - MEX function

            funcs = {}

            def declare_c_func_ptrs(funcs_in):
                global funcs
                # Effectively re-interperet casts.
                funcs['c_raw_to_py'] = \
                    c_raw_to_py_t(funcs_in['c_raw_to_py'])
                funcs['c_py_to_raw'] = \
                    c_py_to_raw_t(funcs_in['c_py_to_raw'])
                        # Marshaling between MATLAB <-> Python
                        funcs['c_mx_raw_to_py_raw'] = \
                            c_raw_to_py_t(funcs_in['c_mx_raw_to_py_raw'])
                        funcs['c_py_raw_to_mx_raw'] = \
                            c_py_to_raw_t(funcs_in['c_py_raw_to_mx_raw'])
                # Calling MEX through type erasure.
                funcs['c_call_matlab'] = \
                    c_call_matlab_t(funcs_in['c_call_matlab'])

            def py_raw_to_py(py_raw):
                return funcs['c_raw_to_py'](py_raw)

            def py_to_py_raw(obj):
                return funcs['c_py_to_raw'](obj)

            def call_matlab(mx_raw_func_handle, nargout, *py_in):
                nargin = len(py_in)
                # Just do a py.list, for MATLAB to convert to a cell arrays.
                py_raw_in = c_py_to_raw(py_in)
                mx_raw_out = mx_array_t * nargout
                py_raw_out = call_matlab_c(mx_raw_func_handle, nargout, py_raw_in)  # Args will be cast to 
                py_out = c_raw_to_py(nargout, mx_out)
                return py_out

                # Is there a way to use mexCallMATLAB to be invoked on Python args?
                # Increase reference counting???
        MATLAB
            py.declare_c_func_ptrs(mex_get_c_func_ptrs())

            py.call_matlab(mex_ml_to_mx_raw(@sin), 1, 0.)

            function py_out = feval_py(f, nout, py_in)
                argin = cell(py_in);
                argout = cell(1, nout);
                [argout{:}] = feval(f, argin{:});
                py_out = py.pass_thru(argout);
            end

TODO: Test passing opaque function pointers from `pybind11`.
    Done.
