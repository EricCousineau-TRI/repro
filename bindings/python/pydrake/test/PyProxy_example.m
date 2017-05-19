% Call "clear classes" if you change your method, then reload the module.
% http://www.mathworks.com/help/matlab/matlab_external/call-modified-python-module.html
pyscratch = pyimport('scratch');
% py.reload(pyscratch);

%% Use Python module directly
fprintf('[ Python Directly ]\n');
pyo = pyscratch.Test('hello');
disp(pyo.get_name())
pyo.set_name('test');
pyo.nparray = [1, 2, 3];
pyo.nparray
% Note that .nparray is now array.array, not numpy.ndarray
pyo.do_stuff();

fprintf('\n\n');

%% Use Python module via PyProxy
fprintf('[ Python via PyProxy ]\n');
mscratch = PyProxy(pyscratch);
% Test = PyProxy.fromPyValue(pyscratch.Test);
% Wrap type

mlo = mscratch.Test('hello');
disp(mlo.get_name());
mlo.set_name('test');
disp(mlo.get_name());
mlo.nparray = [1, 2, 3];
mlo.nparray
% Note that it is still numpy.ndarray
mlo.do_stuff();
