%{
mex_py_proxy('help')
[varargout{:}] = mex_py_proxy(op, varargin)
  op
    'mx_to_mx_raw' Convert mxArray* to uint64 to be passed opaquely.
        [mx_raw] = mex_py_proxy('mx_to_mx_raw', mx)
    'mx_raw_to_mx' Unpack opaque value.
        [mx] = mex_py_proxy('mx_raw_to_mx', mx_raw)
    'get_c_func_ptrs' Get C pointers, using opaque types, to pass to Python.
        [c_func_ptrs_struct] = mex_py_proxy('get_c_func_ptrs')
    'help' Show usage.
%}

make;

%%
clear classes;
MexPyProxy.init();

%%
values = {1, eye(20, 20)};  % Fails
% Be wary of passing temporaries! May be subject to ML gargabe collection.
for i = 1:length(values)
    value = values{i};
%     value = 1;
    mx_raw = mex_py_proxy('mx_to_mx_raw', value);
    mx_value = mex_py_proxy('mx_raw_to_mx', mx_raw, value);
    if ~isequal(value, mx_value)
        value
        mx_value
        warning('Not equal');
    end
end

%%
f = @sin;

MexPyProxy.test_call(f, 1, 0)
