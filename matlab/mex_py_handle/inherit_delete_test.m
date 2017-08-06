clear all; clear classes;
make;
MexPyProxy.init();

ice = pyimport_proxy('inherit_check_py');
PyProxy.reloadPy(ice);

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
