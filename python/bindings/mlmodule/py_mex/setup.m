make;

%%
MexPyProxy.preclear(); % Prevent C functions from getting mixed up if MEX is cleared.
clear all;
clear classes;
MexPyProxy.init();
