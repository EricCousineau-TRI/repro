% py.reload(pyimport('example')); clear classes;

pyexample = pyimport('example');

%%
tup = pyexample.passthrough({4});
tup + {5}

cls = py.type(tup);
% pyAdd = py.getattr(cls, '__add__');
plus(tup, {5})

str = pyexample.passthrough('hello');
str + ' world'

%% 
mtup = PyProxy(tup);
mtup + {5}

%% Test out numpy
np = py.numpy.array([1., 2, 3]);
mlnp = PyProxy(np);

np + 5
np * 5
np / 5
np == 1

% These will all return double matrices (MATLAB), since PyProxy will cast
% them.
mlnp + 5
mlnp * 5
mlnp / 5
mlnp == 1

% This will prevent PyProxy from converting back to double after operators.
custom_array = pyexample.get_custom_array();
% @ref https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
ca = np.view(custom_array);
mlca = PyProxy(ca);

mlca + 5
mlca * 5
mlca / 5
mlca == 1
