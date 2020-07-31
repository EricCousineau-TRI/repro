function [] = startupProject()
% NOTE: This still causes segfaults. Consider locking the MEX function.
prevDir = pwd();
curDir = fileparts(mfilename('fullpath'));
cd(curDir);

%%
make;

%%
MexPyProxy.preclear(); % Prevent C functions from getting mixed up if MEX is cleared.
evalin('base', 'clear all');
evalin('base', 'clear classes');
MexPyProxy.init();

cd(prevDir);

end
