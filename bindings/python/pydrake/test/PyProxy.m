classdef PyProxy < dynamicprops
    properties
        self
    end
    
    methods
        function obj = PyProxy(self)
            % Python form
            obj.self = self;
            
            methods = {'get_name'};
            for i = 1:length(methods)
                method = methods{i};
                pyfunc = @(varargin) obj.self.(method)(varargin{:});
                mfunc = wrapped_py(pyfunc);
                addprop(obj, method);
                obj.(method) = mfunc;
            end
        end
    end
end

function [mfunc] = wrapped_py(f)
mfunc = @(varargin) wrap_py(f, varargin{:});
end

function [m_out] = wrap_py(f, varargin)
map = @(f, C) cellfun(@to_py, C, 'UniformOutput', false);
py_in = map(@to_py, varargin);
py_out = f(py_in{:});
m_out = from_py(py_out);
end

function [py] = to_py(m)
py = m;
end

function [m] = from_py(p)
switch class(p)
    case 'py.str'
        m = char(p);
    case 'py.long'
        m = int64(p);
    case 'py.list'
        m = cell(p);
    otherwise
        m = p;
end
end
