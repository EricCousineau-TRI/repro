classdef PyMxRaw < handle

    properties (Access = protected)
        mx_raw_obj
        
        % Python `MxRaw` object
        % - Need to try and avoid circular references with this design...
        PyRawRef
    end

    methods
        function obj = PyMxRaw(value, disp_value)
            if nargin < 2
                disp_value = sprintf('PyMxRaw:%s', class(value));
            end
            % Create and increment reference count.
            obj.mx_raw_obj = MexPyProxy.mx_to_mx_raw(value);
            % Create Python object.
            obj.PyRawRef = py.py_mex_proxy.MxRaw(obj.mx_raw_obj, disp_value);
            % Decrement reference count after creating here.
            MexPyProxy.mx_raw_ref_decr(obj.mx_raw_obj);
        end
% 
%         function delete(obj)
%             % NOTE: Destruction order, between MATLAB and Python, is very
%             % hairy... This will cause pain.
%             MexPyProxy.mx_raw_ref_decr(obj.mx_raw_obj);
%             value = MexPyProxy.mx_raw_to_mx(obj.mx_raw_obj);
%             fprintf('PyMxRaw: remove %s\n', class(value));
%         end

        function [py_ref] = pyRawRef(obj)
            py_ref = obj.PyRawRef;
        end
    end
end
