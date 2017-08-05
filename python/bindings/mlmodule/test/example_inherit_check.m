ic = pyimport_proxy('pymodule.sub.inherit_check');
ice = pyimport_proxy('example_inherit_check');

%%
cpp = ic.CppExtend()
py = ice.PyExtend()
value = int64(3);

c = cpp.dispatch(value)
py.dispatch(value)
