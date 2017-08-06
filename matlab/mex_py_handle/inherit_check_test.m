make;

%%
MexPyProxy.preclear(); % Prevent C functions from getting mixed up.
clear all; clear classes;
MexPyProxy.init();

%%
ic = pyimport_proxy('pymodule.sub.inherit_check');
ice = pyimport_proxy('inherit_check_py');
PyProxy.reloadPy(ice);
py_simple = pyimport_proxy('simple');
PyProxy.reloadPy(py_simple);
py_pass_thru = py_simple.pass_thru;

%%
cpp = ic.CppExtend()
py = ice.PyExtend(py_pass_thru);
value = int64(4);

cpp.dispatch(value)
py.dispatch(value)

ic.call_method(cpp)
ic.call_method(py)

%%
py = ice.PyExtend(@adder)
py.dispatch(value)

%%
mx = MxExtend();

%%
mx.pure(value)
mx.optional(value)

%%
x = mx.dispatch(value)
% Test calling through a bound method
ic.call_method(mx)

%%
MexPyProxy.erasure()

clear py
mx.free();
clear mx

MexPyProxy.erasure()

%%
mx = InheritCheckMx();
mx.free();

%%
% TODO: Resolve simple reference leak here... From circular references?
MexPyProxy.mx_raw_count()
