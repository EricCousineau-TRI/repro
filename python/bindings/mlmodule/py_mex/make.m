% Per doc:
% https://www.mathworks.com/help/matlab/matlab_external/system-and-configuration-requirements.html
% Print details
pyversion

%%
mex CXXFLAGS='$CXXFLAGS -std=c++14' mex_py_proxy.cpp
