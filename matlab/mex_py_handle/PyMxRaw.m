classdef PyMxRaw < handle
% Class representing a MATLAB extension of a Python object.

    properties (Access = protected)
        PyBaseCls
        PyBaseObj
    end
    
    methods
        function obj = PyMxRaw(pyBaseCls)
            obj.PyBaseCls = pyBaseCls;
            % Construct base proxy and pass object.
            % NOTE: Passing `feval` is just a cheap hack.
            % NOTE: Unwrap so that we can fiddle with values.
            pyCls = PyProxy.toPyValue(obj.PyBaseCls);
            
            function [varargout] = obj_feval_mx_raw(mx_raw_obj, method, varargin)
                MexPyProxy.mx_raw_ref_incr(mx_raw_obj);
                mx_obj = MexPyProxy.mx_raw_to_mx(mx_raw_obj);
                varargout = cell(1, nargout);
                [varargout{:}] = feval(method, mx_obj, varargin{:});
            end
            
            pyFeval = PyProxy.toPyValue(@obj_feval_mx_raw);
            mx_raw_obj = MexPyProxy.mx_to_mx_raw(obj);
            % Don't wrap this in a proxy just yet.
            obj.PyBaseObj = pyCls(mx_raw_obj, pyFeval);
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
%             method_proxy = PyProxy.fromPyValue(method_py);
            fprintf('ml: pyInvokeDirect %s - start\n', method);
%             [varargout{:}] = method_proxy(varargin{:});
            [varargout{:}] = method_py(varargin{:});
            fprintf('ml: pyInvokeDirect %s - finish\n', method);
        end

        % TODO: Make protected
        function [py] = pyObj(obj)
            py = obj.PyBaseObj;
        end
    end
end
