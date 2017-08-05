clear all; clear classes;
make;
MexPyProxy.init();

%% 
fp = pyimport_proxy('pymodule.sub.func_ptr');
py_simple = pyimport_proxy('simple');
PyProxy.reloadPy(py_simple);
py_pass_thru = py_simple.pass_thru;

%%
% First test with just Python
% (C++ calling Python)
fp.call(py_pass_thru)

%% Now with a MATLAB callback
% (C++ calling Python calling MATLAB)
fp.call(@adder)

%% Now try to get Python -> pybind11 (C++) -> Python -> MATLAB -> Python
meta_call = @(f) fp.call(f);

py_simple.call_check(meta_call, @adder)
