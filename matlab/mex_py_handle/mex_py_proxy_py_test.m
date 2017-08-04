MexPyProxy.init();
mex_py = MexPyProxy.py_module();

i1 = mex_py.py_to_py_raw({'a', 'b'})
mex_py.py_raw_to_py(i1)
