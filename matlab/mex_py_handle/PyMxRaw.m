classdef PyMxRaw < handle

    properties (Access = protected)
        mx_raw_obj
        
        % Python `MxRaw` object
        % - This is not a circular reference. Rather, it's that this points
        % to MxRaw, which points to the raw value.
        % (Weird, but that's how it goes when we don't have proper bindings
        % :(  )
        PyRawRef
    end

    methods
        function obj = PyMxRaw(disp_value)
            if nargin < 1
                disp_value = 'ml.PyMxRaw';
            end
            obj.mx_raw_obj = MexPyProxy.mx_to_mx_raw(obj);
            obj.PyRawRef = py.py_mex_proxy.MxRaw(obj.mx_raw_obj, disp_value);
        end

        function delete(obj)
            % NOTE: Destruction order, between MATLAB and Python, is very
            % hairy... This will cause pain.
            MexPyProxy.mx_raw_ref_decr(obj.mx_raw_obj);
        end

        function [py_ref] = pyRawRef(obj)
            py_ref = obj.PyRawRef;
        end
    end
end
