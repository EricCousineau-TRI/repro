clear all; clear classes;
make;
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

c = cpp.dispatch(value)
py.dispatch(value)

%%
py = ice.PyExtend(@adder)
py.dispatch(value)

%%
mx = InheritCheckMx();

%%
mx.pure(value)
mx.optional(value)

%%
x = mx.dispatch(value)
