clear all; clear classes;
make;
MexPyProxy.init();

%%
ice = pyimport_proxy('inherit_check_py');
PyProxy.reloadPy(ice);
pys = pyimport_proxy('simple');
PyProxy.reloadPy(pys);

%%
MexPyProxy.erasure()
mx = InheritCheckMx();

%%
% Hack: For reference counting.
mxc = onCleanup(@() mx.free());
mx.dispatch(int64(1));
MexPyProxy.erasure()
%%
clear mx mxc
MexPyProxy.erasure()

%%
MexPyProxy.erasure()
x = PyMxRaw(1);
MexPyProxy.erasure()
pyx = pys.Store(x)
MexPyProxy.erasure()
clear x
clear pyx
MexPyProxy.erasure()
