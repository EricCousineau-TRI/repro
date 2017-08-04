classdef MexPyProxy
    methods (Static)
        function [] = init()
            addpath(fullfile(pwd, 'py_proxy'));
            % Initialize erasure.
            e = MexPyProxy.erasure(); %#ok<NASGU>
            % Get Python module.
            mex_py = pyimport('mex_py_proxy');
            py.reload(mex_py);
            % Initialize pointers, permit Python to have access to them.
            c_func_ptrs = mex_py_proxy('get_c_func_ptrs');
            mex_py.init_c_func_ptrs(c_func_ptrs);
        end

        function [i] = mx_to_mx_raw(value)
            i = MexPyProxy.erasure().store(value);
        end

        function [value] = mx_raw_to_mx(i)
            value = MexPyProxy.erasure().dereference(i);
        end

        function py_raw_out = mx_feval_py_raw(mx_raw_handle, nout, py_raw_in)
            % feval_py
            % Input: void* representing a Python list containing all input
            % arguments to be converted to MATLAB.
            % Output: void* representing a Python list containing all output.
            py_in = py.py_raw_to_py(py_raw_in);
            mx_in = PyProxy.fromPy(py_in);  % Add depth option?
            mx_out = cell(1, nout);
            mx_handle = MexPyProxy.mx_raw_to_mx(mx_raw_handle);
            [mx_out{:}] = feval(mx_handle, mx_in{:});
            py_out = PyProxy.toPy(mx_out);
            py_raw_out = uint64(py.py_to_py_raw(py_out));
        end
    end

    methods (Static, Access = protected)
        function [out] = erasure()
            persistent e
            if isempty(e)
                e = Erasure();
            end
            out = e;
        end
    end
end
