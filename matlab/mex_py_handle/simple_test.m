make;
clear all; clear classes;

%%
py_mex = MexPyProxy.py_module();
py.reload(py_mex);
MexPyProxy.init();

py.reload(pyimport('simple'));

%%
py_mex.simple();
py.simple.simple();
