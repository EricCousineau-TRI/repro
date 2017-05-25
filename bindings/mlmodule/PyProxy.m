classdef PyProxy % < dynamicprops
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
    
    % TODO(eric.cousineau): Make a separate PyProxy for subsref-able
    % classes.... Numpy makes things as confusing as hell.
    
    properties (Access = protected)
        pySelf
    end
    
    %% Construction
    methods
        function obj = PyProxy(pySelf)
            % Python form
            obj.pySelf = pySelf;
        end
        
        function disp(obj)
            disp('  [PyProxy]');
            disp(obj.pySelf);
        end
        
        function [p] = py(obj)
            p = obj.pySelf;
        end
        
        function r = subsref(obj, S)
            p = PyProxy.getPy(obj);
            s = S(1);
            switch s.type
                case '.'
                    switch s.subs
                        case 'py'
                            r = p;
                        otherwise
                            % Get the proxied value
                            pValue = py.getattr(p, s.subs);
                            r = PyProxy.fromPyValue(pValue);
                    end
                otherwise
                    r = pySubsref(obj, s);
            end
            % Accumulate the remainder
            r = subsref_relaxed(r, S(2:end));
        end
        
        function obj = subsasgn(obj, S, value)
            % Just take scalars for now
            % Get everything leading up to final portion
            penultimate = subsref_relaxed(obj, S(1:end-1));
            % Assumes everything is some sort of proxied thing
            pySubsasgn(penultimate, S(end), value);
        end
    end
    
    methods (Access = protected)
        function r = pySubsref(obj, s)
            p = PyProxy.getPy(obj);
            switch s.type
                case '()'
                    % Assume that we are calling this as a function.
                    r = PyProxy.callPyFunc(p, s.subs{:});
                otherwise
                    error('Unsupported subsref type: %s', s.type);
            end
        end
        
        function pySubsasgn(obj, s, value)
            p = PyProxy.getPy(obj);
            switch s.type
                case '.'
                    pValue = PyProxy.toPyValue(value);
                    py.setattr(p, s.subs, pValue);
                otherwise
                    error('Indexing unsupported');
            end
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
                m = PyProxy(p);
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
                    case 'py.numpy.ndarray'
                        % Do not convert, to preserve memory, yada yada.
                        m = NumPyProxy(p);
                    otherwise
                        % Generate proxy ???
                        warning('Dunno what this is');
                        m = PyProxy(p);
                end
            end
        end
    end
    
    methods (Static, Access = protected)
        function p = getPy(obj)
            % Direct access, do not use subsref overload
            p = builtin('subsref', obj, substruct('.', 'pySelf'));
        end
    end
end
