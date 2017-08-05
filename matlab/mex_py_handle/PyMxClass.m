classdef PyMxClass < PyMxRaw
% Class representing a MATLAB extension of a Python object.
% TODO: Rename this to PyMxClass. Make PyMxRaw handle reference counting.

    properties (Access = protected)
        PyTrampolineCls
        PyTrampolineObj
    end

    methods
        function obj = PyMxClass(pyTrampolineCls)
            % NOTE: See `pybind11`s definition of a trampoline class.
            obj@PyMxRaw();
            
            % This should be proxied.
            obj.PyTrampolineCls = pyTrampolineCls;

            % Construct base proxy and pass object.
            % NOTE: Passing `feval` is just a cheap hack.
            function [varargout] = obj_feval_mx_raw(tobj, method, varargin)
                varargout = cell(1, nargout);
                [varargout{:}] = feval(method, tobj, varargin{:});
            end
            
            % Create object, but don't let the proxy remove the good stuff.
            args = {obj.pyRawRef(), @obj_feval_mx_raw};
            pyargs = cellfun(@PyProxy.toPyValue, args, 'UniformOutput', false);
            pyTrampolineClsRaw = PyProxy.toPyValue(pyTrampolineCls);
            obj.PyTrampolineObj = ...
                pyTrampolineClsRaw(pyargs);
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
            method_py = obj.PyTrampolineObj.(method);
            [varargout{:}] = method_py(varargin{:});
        end

        % TODO: Make protected
        function [py] = pyTrampolineObj(obj)
            py = obj.PyTrampolineObj;
        end
    end
end
