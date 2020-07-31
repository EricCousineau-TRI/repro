function [] = startupProject()
%startupProject Add parent directory to path.

parent = fileparts(fileparts(mfilename('fullpath')));
child = fullfile(parent, 'py_mex');
addpath(parent, child);
pyaddpath(parent, child);

end
