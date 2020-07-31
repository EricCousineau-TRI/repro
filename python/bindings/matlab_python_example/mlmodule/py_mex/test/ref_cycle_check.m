%%
mx = MxExtend();

% Test direct calls on a non-virtual C++ method that calls virtual methods.
value = int32(1);
mx.dispatch(value)

%%
clear mx
fprintf('References, Python -> MX: %d\n', MexPyProxy.mx_raw_count());

