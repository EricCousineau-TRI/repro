make;

%%
MexPyProxy.init();
py_mex = MexPyProxy.py_module();
% mex_py_proxy('py_so_reload');
py_mex.simple();
py.simple.simple();
