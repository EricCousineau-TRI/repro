clear all; clear classes;
make;
MexPyProxy.init();

ice = pyimport_proxy('inherit_check_py');
PyProxy.reloadPy(ice);
pys = pyimport_proxy('simple');
PyProxy.reloadPy(pys);

%%
MexPyProxy.erasure()
mx = InheritCheckMx();
MexPyProxy.erasure()
mx.free();
clear mx
MexPyProxy.erasure()

%%
clear all;
%%
MexPyProxy.erasure()
x = PyMxRaw(1);
MexPyProxy.erasure()
pyx = pys.Store(x)
MexPyProxy.erasure()
clear x
clear pyx
MexPyProxy.erasure()