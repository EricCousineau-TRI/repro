a = py.numpy.array([1, 2, 3]);

get = py.getattr(a, '__getitem__');

get(0)
% https://stackoverflow.com/questions/2936863/python-implementing-slicing-in-getitem
slice = py.slice(1, py.None, py.None); % 1::
get(slice)

set = py.getattr(a, '__setitem__');
set(0, 5)
a
set(slice, [10, 12])
a

%% Ragged
is = py.numpy.array(int32([2, 0]));
get(is)

set(is, [-20, -40])
a

% Can directly access
get(int32([1, 2]))

%% MATLAB Base Case
A = reshape(1:9, 3, 3);

A(:)
A(3, 2)
A([2, 1], 3)

%% "Raw" Python
pyA = py.numpy.eye(3);
pyA.T.flat = py.numpy.arange(1, 10);

% pyA(1)  % Not permitted

get = py.getattr(pyA, '__getitem__');
get({2, 1})
get({[1, 0], 2})

%% Use NumPyProxy
mlA = NumPyProxy(pyA);

mlA(:)
double(mlA(:))

mlA(3, 2)

sub = mlA([2, 1], 3)
sub2 = mlA([1, 2], 3)

% BUG SOMEWHERE: numpy.nditer does not play well with a sliced view of the
% Python array.
double(sub)  % Not working???
double(sub2) % This works. Issue is due to iterator...

%% Assignment
A([2, 1], 3) = 100 * [1, 2]
mlA([2, 1], 3) = 100 * [1, 2]

A(:) = 5
mlA(:) = 5
