classdef PyMxClass < PyMxRaw
% Class representing a MATLAB extension of a Python object.
% TODO: Rename this to PyMxClass. Make PyMxRaw handle reference counting.

    properties (Access = protected)
        PyBaseCls
        PyBaseObj
    end

    methods
        function obj = PyMxClass(pyBaseCls)
            obj@PyMxRaw();
            
            % This should be proxied.
            obj.PyBaseCls = pyBaseCls;

            % Construct base proxy and pass object.
            % NOTE: Passing `feval` is just a cheap hack.
            function [varargout] = obj_feval_mx_raw(tobj, method, varargin)
                varargout = cell(1, nargout);
                [varargout{:}] = feval(method, tobj, varargin{:});
            end
            obj.PyBaseObj = pyBaseCls(obj, @obj_feval_mx_raw);
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
            method_py = py.getattr(obj.PyBaseObj, method);
            [varargout{:}] = method_py(varargin{:});
        end

        % TODO: Make protected
        function [py] = pyObj(obj)
            py = obj.PyBaseObj;
        end
    end
end
