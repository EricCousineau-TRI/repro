classdef PyProxy < dynamicprops
    % Quick and dirty mechanism to wrap object instance or module to marshal MATLAB data types
    % This might be more easily solved if the MATLAB:Python bindings were
    % more versatile, e.g., working with matrices.
    
    % TODO(eric.cousineau): Try not to wrap each instance returned, but
    % rather wrap the class (to minimize on memory overhead).
    % Presently, each instance is wrapped.
    
    % TODO(eric.cousineau): This won't play nicely with a MATLAB double
    % matrix that is meant to be a Python double list.
    % Solution is to use a cell() matrix.
    
    % TODO(eric.cousineau): Add operator overloads?
    
    properties (Access = protected)
        pySelf
    end
    
    methods
        function obj = PyProxy(pySelf)
            % Python form
            obj.pySelf = pySelf;
            
            pyPropNames = py.dir(pySelf);
            for i = 1:length(pyPropNames)
                propName = char(pyPropNames{i});
                % Skip MATLAB invalid / Python private members
                if length(propName) >= 1 && strcmp(propName(1), '_')
                    continue;
                end
                pyValue = py.getattr(obj.pySelf, propName);
                mlProp = addprop(obj, propName);
                % Add getter (which will work for functions / methods)
                mlProp.GetMethod = @(obj) ...
                    PyProxy.fromPyValue(py.getattr(obj.pySelf, propName));
                % Hide setter if it's a function 
                if PyProxy.isPyFunc(pyValue)
                else
                    % Permit setting
                    % TODO(eric.cousineau): See if there is a way to check
                    % if this is a read-only or write-only setter?
                    % https://docs.python.org/2/library/functions.html#property
                    mlProp.SetMethod = @(obj, value) ...
                        py.setattr(obj.pySelf, propName, PyProxy.toPyValue(value));
                end
            end
        end
        
        function disp(obj)
            disp('  [PyProxy]');
            disp(obj.pySelf);
        end
    end
    
    methods (Static)
        function out = isPyFunc(p)
            % @ref http://stackoverflow.com/questions/624926/how-to-detect-whether-a-python-variable-is-a-function
            out = py.hasattr(p, '__call__');
        end
        
        function out = isPyClass(p)
            % @ref http://stackoverflow.com/questions/395735/how-to-check-whether-a-variable-is-a-class-or-not
            out = py.inspect.isclass(p);
        end
        
        function out = isPyWrappable(p)
            % Wrap if function (with __call__) or if class type (to wrap
            % constructor)
            out = PyProxy.isPyFunc(p) || PyProxy.isPyClass(p);
        end
        
        function mfunc = wrapPyFunc(f)
            %wrapPyFunc Wrap Python function as Matlab function, to handle
            %marshalling arguments.
            mfunc = @(varargin) PyProxy.callPyFunc(f, varargin{:});
        end
        
        function [m_out] = callPyFunc(f, varargin)
            map = @(f, C) cellfun(f, C, 'UniformOutput', false);
            py_in = map(@PyProxy.toPyValue, varargin);
            py_out = f(py_in{:});
            m_out = PyProxy.fromPyValue(py_out);
        end
        
        function [p] = toPyValue(m)
            switch class(m)
                case 'double'
                    % Will need to relax Eigen types to accept scalar
                    % doubles to initialize as 1x1.
                    % Use C++ py_relax_overload shindig.
                    if isscalar(m)
                        p = m;
                    else
                        p = matpy.mat2nparray(m);
                    end
                case 'PyProxy'
                    p = PyProxy.pySelf;
                otherwise
                    % Defer to MATLAB:Python.
                    p = m;
            end
        end
        
        function [m] = fromPyValue(p)
            if PyProxy.isPyWrappable(p)
                m = PyProxy.wrapPyFunc(p);
            else
                switch class(p)
                    case 'py.str'
                        m = char(p);
                    case 'py.long'
                        m = int64(p);
                    case 'py.list'
                        m = cell(p);
                    case 'py.numpy.ndarray'
                        m = matpy.nparray2mat(p);
                    case 'py.NoneType'
                        m = [];
                    case 'py.module'
                        % This should be proxy-able
                        % Use this SPARINGLY
                        m = PyProxy(p);
                    otherwise
                        % Generate proxy
                        m = PyProxy(p);
                end
            end
        end
    end
end
