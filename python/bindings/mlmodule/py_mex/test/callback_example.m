% %%
% setup;

%% 
func_ptr_py = pyimport_proxy('pymodule.sub.func_ptr');
simple_py = pyimport_proxy('simple');
PyProxy.reloadPy(simple_py);
pass_thru_py = simple_py.pass_thru;

%% Test modalities
% C++ calling Python
func_ptr_py.call_cpp(pass_thru_py)
% C++ calling Python calling C++
func_ptr_py.call_cpp(func_ptr_py.func_cpp)
% C++ calling Python calling MATLAB
func_ptr_py.call_cpp(@adder)

%% Excessive example, pushing stuff together.
meta_call = @(f) func_ptr_py.call_cpp(f);
simple_py.call_check(meta_call, pass_thru_py(@adder))
