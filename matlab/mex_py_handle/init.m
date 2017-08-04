function init()
py.mx_py.declare_c_func_ptrs(mex_py_proxy('get_c_func_ptrs'));
% py.mx_py.call_matlab(mex_py_proxy('mx_to_mx_raw', @sin), 1, 0.)
end
