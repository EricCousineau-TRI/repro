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

value = 1;
mx_raw = mex_py_proxy('mx_to_mx_raw', value);
mx = mex_py_proxy('mx_raw_to_mx', mx_raw);
