% %%
% setup;

%%
inherit_check_py = pyimport_proxy('pymodule.sub.inherit_check');
inheritance_example_py = pyimport_proxy('inheritance_example');

PyProxy.reloadPy(inherit_check_py);
PyProxy.reloadPy(inheritance_example_py);

%%
value = int64(4);

cpp = inherit_check_py.CppExtend();
py = inheritance_example_py.PyExtend();
mx = MxExtend();

% Test direct calls on a non-virtual C++ method that calls virtual methods.
cpp.dispatch(value)
py.dispatch(value)
mx.dispatch(value)

% Test calling through a bound method
inherit_check_py.call_method(cpp)
inherit_check_py.call_method(py)
inherit_check_py.call_method(mx)

%%
% Unfortunately, reference counting is hacky. Manually do a `gc` cycle
% counting step, otherwise we accumulate references.
mx.free();

% There should be one reference, for `feval` used to dispatch Python ->
% MATLAB virtual calls.
fprintf('References, Python -> MX: %d\n', MexPyProxy.mx_raw_count());
