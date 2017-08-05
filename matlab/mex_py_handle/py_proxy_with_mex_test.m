%%
make;
clear all;
clear classes; %#ok<CLCLS>
MexPyProxy.init();

%%
py_simple = pyimport_proxy('simple');
py.reload(PyProxy.toPyValue(py_simple));

%%
f = @(x) x / pi;
a = py_simple.call_check(f, pi / 4)
b = py_simple.call_check(f, [pi / 4, 10 * pi])
