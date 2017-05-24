classdef PyProxy < dynamicprops
    % Quick and dirty mechanism to wrap object instance or module to marshal MATLAB data types
    % This might be more easily solved if the MATLAB:Python bindings were
    % more versatile, e.g., working with matrices.
    
    % TODO(eric.cousineau): Try not to wrap each instance returned, but
    % rather wrap the class (to minimize on memory overhead).
    % Presently, each instance is wrapped.
    
    % TODO(eric.cousineau): See if there is a better way to control when an
    % instance is converted (e.g., "expr = ~expr", if "expr" is a numpy
    % array...).
    
    % TODO(eric.cousineau): This won't play nicely with a MATLAB double
    % matrix that is meant to be a Python double list.
    % Solution is to use a cell() matrix.
    
    properties (Access = protected)
        pySelf
    end
    
    %% Construction
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
        
        function [p] = py(obj)
            p = obj.pySelf;
        end
    end
    
    %% Operator overloads
    % @ref https://www.mathworks.com/help/matlab/matlab_oop/implementing-operators-for-your-class.html
    methods
        % How to handle concatenation?
        function r = plus(a, b)
            % Quick hack for now - delegte overloads to Python.
            r = PyProxy.callPyFunc(@plus, a, b);
        end
        function r = minus(a, b)
            r = PyProxy.callPyFunc(@minus, a, b);
        end
        function r = uminus(a)
            r = PyProxy.callPyFunc(@uminus, a);
        end
        function r = uplus(a)
            r = PyProxy.callPyFunc(@uplus, a);
        end
        function r = times(a, b)
            r = PyProxy.callPyFunc(@times, a, b);
        end
        function r = mtimes(a, b)
            r = PyProxy.callPyFunc(@mtimes, a, b);
        end
        function r = rdivide(a, b)
            r = PyProxy.callPyFunc(@rdivide, a, b);
        end
        function r = mrdivide(a, b)
            % This may be ambiguous. Still gonna leave it up to MATLAB
            % Python bridge to figure it out.
            r = PyProxy.callPyFunc(@mrdivide, a, b);
        end
        function r = power(a, b)
            r = PyProxy.callPyFunc(@power, a, b);
        end
        function r = mpower(a, b)
            % Same as mrdivide
            r = PyProxy.callPyFunc(@mpower, a, b);
        end
        % Logical
        function r = lt(a, b)
            r = PyProxy.callPyFunc(@lt, a, b);
        end
        function r = gt(a, b)
            r = PyProxy.callPyFunc(@gt, a, b);
        end
        function r = le(a, b)
            r = PyProxy.callPyFunc(@le, a, b);
        end
        function r = ge(a, b)
            r = PyProxy.callPyFunc(@ge, a, b);
        end
        function r = ne(a, b)
            r = PyProxy.callPyFunc(@ne, a, b);
        end
        function r = eq(a, b)
            r = PyProxy.callPyFunc(@eq, a, b);
        end
        function r = and(a, b)
            r = PyProxy.callPyFunc(@and, a, b);
        end
        function r = or(a, b)
            r = PyProxy.callPyFunc(@or, a, b);
        end
        function r = not(a)
            % This cannot be overridden in Python.
            % For now, we can delegate to numpy.logical_not and cross our
            % fingers.
            r = PyProxy.callPyFunc(@py.numpy.logical_not, a);
        end
        function r = reshape(obj, varargin)
            % Call Python version, not MATLAB
            f = @(varargin) obj.pySelf.reshape(varargin{:});
            r = PyProxy.callPyFunc(f, varargin{:});
        end
        % Confusing...
        %{
        % Referencing
        function r = subsref(obj, ss)
            s = ss(1); % Uhhh...???
            switch s.type
                case '.'
                    % Member access
                    % Figure out more elegenat solution:
                    % https://www.mathworks.com/help/matlab/ref/subsref.html
                    r = obj.(s.subs);
                case '()'
                    % Hack. Could figure out slicing / 2D stuff later.
                    r = obj.item(int64(s.subs{1}) - 1);
            end
            if length(ss) == 2
                % Err...
                assert(strcmp(ss(2).type, '()'));
                % Call function
                assert(isempty(ss(2).subs));
                r = r();
            end
        end
        % No assignment.
        %}
    end
    
    %% Helper methods
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
                    % Use C++ py_relax_overload and RelaxMatrix<> to permit
                    % Eigen matrices to be initialized by scalars from
                    % Python.
                    if isscalar(m)
                        p = m;
                    else
                        p = matpy.mat2nparray(m);
                    end
                case 'PyProxy'
                    p = m.pySelf;
                otherwise
                    % Defer to MATLAB:Python.
                    p = m;
            end
        end
        
        function [out] = isPy(p)
            cls = class(p);
            if length(cls) >= 3 && strcmp(cls(1:3), 'py.')
                out = true;
            else
                out = false;
            end
        end
        
        function [m] = fromPyValue(p)
            if ~PyProxy.isPy(p)
                m = p;
            elseif PyProxy.isPyWrappable(p)
                m = PyProxy.wrapPyFunc(p);
            else
                cls = class(p);
                switch cls
                    case 'py.bool'
                        m = logical(p);
                    case 'py.str'
                        m = char(p);
                    case 'py.long'
                        m = int64(p);
                    case 'py.list'
                        m = cell(p);
                    case 'py.NoneType'
                        m = [];
                    case 'py.module'
                        % This should be proxy-able
                        % Use this SPARINGLY
                        m = PyProxy(p);
                    otherwise
                        good = false;
                        if strcmp(cls, 'py.numpy.ndarray') %#ok<STISA>
                            helper = pyimport('proxy_helper'); % TODO: Use PYTHONPATH
                            % Ensure that this is an integral type
                            if helper.np_is_arithmetic(p)
                                good = true;
                                m = matpy.nparray2mat(p);
                            end
                        end
                        % TODO: Wrapping class as function does not permit
                        % accessing constant values...
                        if ~good
                            % Generate proxy
                            m = PyProxy(p);
                        end
                end
            end
        end
    end
end
