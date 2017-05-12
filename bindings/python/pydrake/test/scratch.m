m = pyimport('scratch');
py.reload(m);

a = m.stuff_py()
b = m.stuff_py(pyargs('is_int', true))

c = m.stuff_numpy()

% Will throw an error
m.stuff_matlab()
