ic = pyimport_proxy('pymodule.sub.inherit_check');
ice = pyimport_proxy('example_inherit_check');

PyProxy.reloadPy(ic);
PyProxy.reloadPy(ice);

%%
value = int64(3);

cpp = ic.CppExtend()
c = cpp.dispatch(value)

%%
py = ice.PyExtend(ice.pass_thru)
py.dispatch(value)
