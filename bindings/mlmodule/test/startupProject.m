function [] = startupProject()
%startupProject Add parent directory to path.

addpath(fileparts(fileparts(mfilename('fullpath'))));

end
