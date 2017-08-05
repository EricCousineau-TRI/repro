ic = pyimport_proxy('pymodule.sub.inherit_check');
ice = pyimport_proxy('example_inherit_check');

PyProxy.reloadPy(ic);
PyProxy.reloadPy(ice);

%%
cpp = ic.CppExtend()
%%
py = ice.PyExtend() % ice.pass_thru)
value = int64(3);

c = cpp.dispatch(value)
py.dispatch(value)
