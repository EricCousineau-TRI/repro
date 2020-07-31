%%
make;
clear classes; %#ok<CLCLS>
MexPyProxy.init();

%%
x = MexPyProxy.test_call(@sin, pi / 4)
assert(abs(x - 1/sqrt(2)) < eps);

%%
py_mex = MexPyProxy.py_module();

value = {'a', 'b'};
i1 = py_mex.py_to_py_raw(value)
value_out = py_mex.py_raw_to_py(i1)
assert(isequal(py.simple.pass_thru(value), value_out));
