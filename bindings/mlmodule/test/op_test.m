pyexample = pyimport('example');

tup = pyexample.passthrough({4});
tup + {5}

cls = py.type(tup);
% pyAdd = py.getattr(cls, '__add__');
plus(tup, {5})

str = pyexample.passthrough('hello');
str + 'you'

%% 
mtup = PyProxy(tup);
mtup + {5}
