%%
x = py.numpy.random.random(([4, 4]));
% % Printing this throws an error
% y = py.numpy.matlib.matrix([1, 2, 3]);

%%
pyscratch = pyimport('scratch');
py.reload(pyscratch);
% Call "clear classes" if you get instance method type mismatches.
% http://www.mathworks.com/help/matlab/matlab_external/call-modified-python-module.html

a = pyscratch.stuff_py()
b = pyscratch.stuff_py(pyargs('is_int', true))
c = pyscratch.stuff_numpy()

% % Will throw an error
% m.stuff_matlab()
