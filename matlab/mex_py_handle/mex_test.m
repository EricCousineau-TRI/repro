%%
make;
clear classes; %#ok<CLCLS>
MexPyProxy.init();

%%
x = MexPyProxy.test_call(@sin, pi / 4)

%%
x = MexPyProxy.test_call(@bad_func, pi / 4)
