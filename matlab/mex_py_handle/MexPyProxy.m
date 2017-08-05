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
            % feval_py
            % Input: py_raw_t representing a Python list containing all input
            % arguments to be converted to MATLAB.
            % Output: py_raw_t representing a Python list containing all output.
            disp('ml: mx_feval_py_raw');
            disp({mx_raw_handle, nout, py_raw_in});
%             py_mex = MexPyProxy.py_module();
            disp('ml: mx_raw_to_mx');
            mx_handle = MexPyProxy.mx_raw_to_mx(mx_raw_handle);
            disp('ml:');
            disp(mx_handle);
%             disp('Convert py');
%             py.simple.simple();
%             py_in = py_mex.py_raw_to_py(py_raw_in);
%             disp('Have py');
%             disp(py_in);
%             mx_in = PyProxy.fromPyValue(py_in);  % Add depth option?
%             disp(mx_in);
%             mx_out = cell(1, nout);
%             
%             disp('feval');
%             [mx_out{:}] = feval(mx_handle, mx_in{:});
%             disp(mx_out);
%             
%             py_out = PyProxy.toPyValue(mx_out);
%             py_raw_out = uint64(py.py_to_py_raw(py_out));
            disp('ml: done');
        end
        
        function [] = test_call(mx_handle, nout, varargin)
            py_mex = MexPyProxy.py_module();
            
            mx_raw_handle = MexPyProxy.mx_to_mx_raw(mx_handle);
            py_mex.mx_raw_feval_py(mx_raw_handle, nout, varargin{:});
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
                py_mex = pyimport('mex_py_proxy');
                py.reload(py_mex);
            end
            out = py_mex;
        end
    end
end
