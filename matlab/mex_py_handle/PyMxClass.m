classdef PyMxClass < handle
% Class representing a MATLAB extension of a Python object.

    properties (Access = protected)
%         PyMxRawObj 
        PyTrampolineCls
        PyTrampolineObj
    end

    methods
        function obj = PyMxClass(pyTrampolineCls)
%             % NOTE: See `pybind11`s definition of a trampoline class.
%             obj@PyMxRaw();
%             
            % This should be proxied.
            obj.PyTrampolineCls = pyTrampolineCls;

            % Construct base proxy and pass object.
            % NOTE: Passing `feval` is just a cheap hack.
            function [varargout] = obj_feval_mx_raw(tobj, method, varargin)
                varargout = cell(1, nargout);
                [varargout{:}] = feval(method, tobj, varargin{:});
            end
            
            % How to do proper reference counting with trampoline classes?
            
            % Create object, but don't let the proxy remove the good stuff.
            % Let trampoline handle conversion stuff.
            py_mx_raw = PyMxRaw(obj);
            args = {py_mx_raw, @obj_feval_mx_raw};
            % WARNING: This looks like a circular reference...
            pyargs = cellfun(@PyProxy.toPyValue, args, 'UniformOutput', false);
            pyTrampolineClsRaw = PyProxy.toPyValue(pyTrampolineCls);
            obj.PyTrampolineObj = ...
                pyTrampolineClsRaw(pyargs{:});
        end
        
        function [] = free(obj)
            % TODO: Since there is a circular reference, objects must be
            % explicitly freed... Is there a better way to synchronize
            % reference counting for a parent/child object pair (if they
            % can't represent the same slot in one reference counting
            % mechanism?)
%             mx_raw = int64(obj.PyTrampolineObj.mx_obj.mx_raw);
%             MexPyProxy.mx_raw_ref_decr(mx_raw);
            obj.PyTrampolineObj.free();
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

        % TODO: Make protected
        function [py] = pyTrampolineObj(obj)
            py = obj.PyTrampolineObj;
        end
    end
end
