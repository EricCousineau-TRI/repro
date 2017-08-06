h = @cos;
x =0.6;
y = simplefunction(h,x)

py_func = @py.stuff.test;
z = simplefunction(py_func, x)

py_value = py.long(2)
% Can pass the value in, but cannot do anything with it.
simplefunction(py_func, py_value)
