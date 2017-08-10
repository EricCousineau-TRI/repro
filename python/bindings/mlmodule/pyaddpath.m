function [] = pyaddpath(varargin)
% Place path at the front of sys.path and ${PYTHONPATH}.
path = py.sys.path;
for i = length(varargin):-1:1
    p = varargin{i};
    if path.count(p) == 1
        path.remove(p);
    end
    path.insert(int64(0), p);
end
% Export to PYTHONPATH as well.
ps = cellfun(@char, cell(path), 'UniformOutput', false);
setenv('PYTHONPATH', strjoin(ps, pathsep));
end
