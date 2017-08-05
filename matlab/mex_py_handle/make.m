%%
% Per doc:
% https://www.mathworks.com/help/matlab/matlab_external/system-and-configuration-requirements.html
% Get Python:
assert(strcmp(pyversion(), '2.7'));
% Print details
pyversion

%%
mex -v -I/usr/include/python2.7 -lpython2.7 -ldl CXXFLAGS='$CXXFLAGS -fpermissive' mex_py_proxy.cpp
% Disable warning: -w
