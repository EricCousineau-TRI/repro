clear all; clear classes;
make;
MexPyProxy.init();

%% 
fp = pyimport_proxy('pymodule.sub.func_ptr');
py_simp = @py.simple.pass_thru;

%%
% First test with just Python
% (C++ calling Python)
fp.call(py_simp)

%% Now with a MATLAB callback
% (C++ calling Python calling MATLAB)
fp.call(@adder)
