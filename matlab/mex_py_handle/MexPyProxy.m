classdef MexPyProxy
    methods (Static)
        function [] = init()
            addpath(fullfile(pwd, 'py_proxy'));
            % Initialize erasure.
            e = MexPyProxy.erasure(); %#ok<NASGU>
            % Get Python module.
            py_mex = MexPyProxy.py_module();
            % Initialize pointers, permit Python to have access to them.
            c_func_ptrs = mex_py_proxy('get_c_func_ptrs');
            py_mex.init_c_func_ptrs(c_func_ptrs);
        end

        function [i] = mx_to_mx_raw(value)
            i = MexPyProxy.erasure().store(value);
        end

        function [value] = mx_raw_to_mx(i)
            value = MexPyProxy.erasure().dereference(i);
        end

        function py_raw_out = mx_feval_py_raw(mx_raw_handle, nout, py_raw_in)
            % This will be called by Python, which will have been called by
            % MATLAB.
            py_mex = MexPyProxy.py_module();
            % Marshal MATLAB function handle from C.
            mx_handle = MexPyProxy.mx_raw_to_mx(mx_raw_handle);
            py_in = py_mex.py_raw_to_py(py_raw_in);
            % Convert each argument.
            mx_in = cellfun(@PyProxy.fromPyValue, cell(py_in), ...
                'UniformOutput', false);
            % Call the MATLAB method.
            mx_out = cell(1, nout);
            [mx_out{:}] = feval(mx_handle, mx_in{:});
            % Marshal back to C-friendly Python types.
            py_out = cellfun(@PyProxy.toPyValue, mx_out, ...
                'UniformOutput', false);
            py_raw_out = uint64(py_mex.py_to_py_raw(py_out));
        end

        function [varargout] = test_call(mx_handle, varargin)
            py_mex = MexPyProxy.py_module();
            mx_raw_handle = MexPyProxy.mx_to_mx_raw(mx_handle);
            py_out = py_mex.mx_raw_feval_py(mx_raw_handle, nargout, varargin{:});
            varargout = cellfun(@PyProxy.fromPyValue, cell(py_out), ...
                'UniformOutput', false);
        end
    end

    methods (Static) %, Access = protected)
        function [out] = erasure()
            persistent e
            if isempty(e)
                e = Erasure();
            end
            out = e;
        end
        function [out] = py_module()
            persistent py_mex
            if isempty(py_mex)
                py_mex = pyimport('py_mex_proxy');
                py.reload(py_mex);
            end
            out = py_mex;
        end
    end
end
