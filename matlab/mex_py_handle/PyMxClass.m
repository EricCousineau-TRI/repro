classdef PyMxClass < handle
% Class representing a MATLAB extension of a Python object.

    properties (Access = protected)
        PyTrampolineCls
        PyTrampolineObj
    end

    methods
        function obj = PyMxClass(pyTrampolineCls, varargin)
            % NOTE: See `pybind11`s definition of a trampoline class.
            % This should be proxied.
            obj.PyTrampolineCls = pyTrampolineCls;
            % How to do proper reference counting with trampoline classes?
            
            % Create object, but don't let the proxy remove the good stuff.
            % Let trampoline handle conversion stuff.
            % WARNING: This is a circular reference...
            py_mx_obj = PyMxRaw(obj);
            pyargin = cellfun(@PyProxy.toPyValue, varargin, 'UniformOutput', false);
            pyTrampolineClsRaw = PyProxy.toPyValue(pyTrampolineCls);
            obj.PyTrampolineObj = ...
                pyTrampolineClsRaw(pyargin{:}, ...
                        pyargs('mx_obj', py_mx_obj.pyRawRef()));
        end
        
        function [] = free(obj)
            % TODO: Since there is a circular reference, objects must be
            % explicitly freed... Is there a better way to synchronize
            % reference counting for a parent/child object pair (if they
            % can't represent the same slot in one reference counting
            % mechanism?)
            free = py.getattr(obj.PyTrampolineObj, '_mx_free');
            free();
            obj.PyTrampolineObj = [];
        end
        
        function [varargout] = pyInvokeVirtual(obj, method, varargin)
            % TODO: Check method type, then invoke.
            % For now, just execute.
            assert(~strcmp(method, 'pyInvokeVirtual'));
            varargout = cell(1, nargout);
            [varargout{:}] = feval(method, obj, varargin{:});
        end

        function [varargout] = pyInvokeDirect(obj, method, varargin)
            assert(~strcmp(method, 'pyInvokeDirect'));
            varargout = cell(1, nargout);
            method_py = py.getattr(obj.PyTrampolineObj, method);
            method_proxy = PyProxy.fromPyValue(method_py);
            [varargout{:}] = method_proxy(varargin{:});
        end

        function [py] = pyTrampolineObj(obj)
            py = obj.PyTrampolineObj;
        end
    end
end
