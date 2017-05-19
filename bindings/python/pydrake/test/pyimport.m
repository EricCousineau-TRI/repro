function [m] = pyimport(varargin)
%pyimport Explicitly import a Python module that may not be accessible via
%  the PY helper.
%
% Example:
%
%  >> tb = pydrake.typebinding               % Will produce an error
%  >> tb = pyimport('pydrake.typebinding')   % No error

if length(varargin) == 1
    % Attempt to split by '.' separator
    full = varargin{1};
    parts = strsplit(full, '.');
else
    parts = varargin;
end
m = py.importlib.import_module(parts{1});
for i = 2:length(parts)
    cur = parts{i};
    m = m.(cur);
end
end
