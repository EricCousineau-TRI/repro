make;

%%
MexPyProxy.init();
mex_py = MexPyProxy.py_module();
% mex_py_proxy('py_so_reload');
mex_py.simple();
