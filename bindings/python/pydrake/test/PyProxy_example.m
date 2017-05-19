% Call "clear classes" if you change your method, then reload the module.
% http://www.mathworks.com/help/matlab/matlab_external/call-modified-python-module.html
pyscratch = pyimport('scratch');
% py.reload(pyscratch);

%% Use Python module directly
fprintf('[ Python Directly ]\n');
pyo = pyscratch.Test('hello');
disp(pyo.get_name())
pyo.set_name('test');
pyo.nparray = [5, 10, 15];
% pyo.nparray(3) = 20; % Invalid - MATLAB does not permit indexing into Python object
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
mlo.nparray = [5, 10, 15];
mlo.nparray(3) = 20; % Permitted - getter/setter combo in MATLAB permits natural indexing (but slow due to copies)
mlo.nparray
% Note that it is still numpy.ndarray
mlo.do_stuff();
