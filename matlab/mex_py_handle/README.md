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
            # int call_matlab_c(mxArray* func_handle, int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[])
            call_matlab_c_t = CFUNCTYPE(c_int, mx_array_t, c_int, mx_array_t_p, c_int, mx_array_t_p)
            # int mx_to_py_t(int n, mxArray* mx_in[], void* py_out[])
            mx_to_py_t = ctypes.CFUNCTYPE(c_int, c_int, mx_array_t_p, POINTER(py_object))
            # int 

            # py - Python
            # py_raw - Raw Python points (py_object, c_void_p)
            # ml, mx - MATLAB
            # mx_raw - void* representing mxArray*
            # mex - MEX function


            funcs = {}
            funcs_raw = {}
            handles = {}

            def declare_mex_funcs(funcs_in):
                global funcs funcs_raw
                funcs_raw = funcs_in
                funcs['call_matlab_c'] = \
                    call_matlab_c_t(funcs_in['call_matlab_c'])
                funcs['mx_to_py'] = \
                    mx_to_py_t(funcs_in['mx_to_py'])
                funcs['py_to_mx'] = \
                    py_to_mx_t(funcs_in['py_to_mx'])

            def declare_mx_func_handles(handles_in):
                global handles
                handles = handles_in

            def raw_to_py(n, raw):
                raw_a = cast(raw, c_void_p)
                py = [None] * n
                for i in xrange(n):
                    py[i] = deref_py(raw[i])
                return py

            def deref_py(raw):
                # This should be of type c_void_p
                # Recover original Python object reference.
                obj = cast(raw, py_object).value
                return obj

            def mx_raw_to_py(mx_in):
                

            def py_to_mx(py_in):

            def call_matlab(mx_func_handle, nargout, *py_in):
                nargin = len(py_in)
                mx_in = py_to_mx(py_in)
                mx_out = mx_array_t * nargout
                call_matlab_c(mx_func_handle, len(in), in, nargout, mx_out)  # Args will be cast to 
                py_out = mx_to_py(nargout, mx_out)

                # Is there a way to use mexCallMATLAB to be invoked on Python args?
                # Increase reference counting???
        MATLAB
            py.declare_mex_func_ptrs(mex_get_func_ptrs())
            s = struct();
            s.mx_raw_to_ml = @mex_mx_raw_to_ml  % f(n, mx)
            s.ml_to_mx_raw = @mex_ml_to_mx_raw
            py.declare_func_handles(structfun(@mex_get_mx_array, s))

            py.call_matlab(mex_ml_to_mx_raw(@sin), 1, 0.)

            function py_mx = py_raw_to_py_mx(n, py_raw)
                % Get MATLAB object representing Python stuff.
                py_mx = py.raw_to_py(n, py_raw)
            end


TODO: Test passing opaque function pointers from `pybind11`.
