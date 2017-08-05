ic = pyimport_proxy('pymodule.sub.inherit_check');
ice = pyimport_proxy('example_inherit_check');

%%
cpp = ic.CppExtend()
py = ice.PyExtend()
value = 'a';

c = cpp.dispatch(value)
py.dispatch(value)
