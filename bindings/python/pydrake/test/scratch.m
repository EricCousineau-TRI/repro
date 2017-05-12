%%
x = py.numpy.random.random([4, 4]);
y = py.numpy.matlib.matrix([1, 2, 3]);

%%
m = pyimport('scratch');
py.reload(m);

a = m.stuff_py()
b = m.stuff_py(pyargs('is_int', true))

c = m.stuff_numpy()

% Will throw an error
m.stuff_matlab()
