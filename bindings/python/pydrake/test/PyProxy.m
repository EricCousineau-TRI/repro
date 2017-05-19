classdef PyProxy < dynamicprops
    properties
        self
    end
    
    methods
        function obj = PyProxy(self)
            % Python form
            obj.self = self;
            
            pyPropNames = py.dir(self);
            for i = 1:length(pyPropNames)
                propName = char(pyPropNames{i});
                % Skip MATLAB invalid / Python private members
                if length(propName) >= 1 && strcmp(propName(1), '_')
                    continue;
                end
                pyValue = py.getattr(obj.self, propName);
                mlProp = addprop(obj, propName);
                if PyProxy.isPyFunc(pyValue)
                    mlFunc = PyProxy.wrapPyFunc(pyValue);
                    obj.(propName) = mlFunc;
                else
                    % Add property with getter / setter
                    mlProp.GetMethod = @(self) ...
                        PyProxy.fromPyValue(py.getattr(self, propName));
                    mlProp.SetMethod = @(self, value) ...
                        py.setattr(self, propName, PyProxy.toPyValue(value));
                end
            end
        end
    end
    
    methods (Static)
        function out = isPyFunc(f)
            % @ref http://stackoverflow.com/questions/624926/how-to-detect-whether-a-python-variable-is-a-function
            out = py.hasattr(f, '__call__');
        end
        
        % If able to wrap classes in general...
        % @ref http://stackoverflow.com/questions/395735/how-to-check-whether-a-variable-is-a-class-or-not
        
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
                    p = matpy.mat2nparray(m);
                case 'PyProxy'
                    p = PyProxy.self;
                otherwise
                    p = m;
            end
        end
        
        function [m] = fromPyValue(p)
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
                otherwise
                    % Generate proxy
                    m = PyProxy(p);
            end
        end
    end
end
