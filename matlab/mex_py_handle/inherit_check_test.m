clear all; clear classes;
make;
MexPyProxy.init();

%%
ic = pyimport_proxy('pymodule.sub.inherit_check');
ice = pyimport_proxy('inherit_check_py');
PyProxy.reloadPy(ice);

%%
cpp = ic.CppExtend()
py = ice.PyExtend()
value = int64(4);

c = cpp.dispatch(value)
py.dispatch(value)

%%
mx = InheritCheckMx();

mx.pure(value)
mx.optional(value)

%%
mx.dispatch(value)
