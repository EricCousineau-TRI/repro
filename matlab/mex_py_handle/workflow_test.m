%%
make;
clear all;
clear classes; %#ok<CLCLS>
MexPyProxy.init();

%%
py_simple = pyimport_proxy('simple');
PyProxy.reloadPy(py_simple);

%%
f = @(x) x / pi;
a = py_simple.call_check(f, pi / 4)
b = py_simple.call_check(f, [pi / 4, 10 * pi])
fs = @(s) [s, ' world'];
c = py_simple.call_check(fs, 'Hello')
d = py_simple.call_check(fs, 'What a')

fprintf('Ref count: %d\n', MexPyProxy.mx_raw_count());

%%
% finv = @(A, b) A \ b;
e = py_simple.call_check(@mldivide, 2 * eye(2), [2, 4]')

fprintf('Ref count: %d\n', MexPyProxy.mx_raw_count());

%%
% NOTE: This won't work, since `struct` will turn into a `dict` in Python,
% but not on the way out (since the Python dict can have invalid struct
% fields).
py_simple.call_check(@fieldnames, struct('test', 1))

%%
clear objs
objs = cell(1, 2);
for i = 1:2
    objs{i} = py_simple.Obj(@sin);
    for j = 1:2
        objs{i}.call(pi / 4)
    end
end
fprintf('Ref count: %d\n', MexPyProxy.mx_raw_count());

%% Show number of live references
fprintf('Ref count: %d\n', MexPyProxy.mx_raw_count());
clear objs
fprintf('Ref count: %d\n', MexPyProxy.mx_raw_count());

%%
%{
<MxFunc: @(x)x/pi>

a =

    0.2500

<MxFunc: @(x)x/pi>

b = 

  [NumPyProxy]
[[  0.25  10.  ]]
<MxFunc: @(s)[s,' world']>

c =

Hello world

<MxFunc: @(s)[s,' world']>

d =

What a world

<MxFunc: mldivide>

e = 

  [NumPyProxy]
[[ 1.]
 [ 2.]]
<MxFunc: fieldnames>

ans =

  1×0 empty cell array

py: Store <MxFunc: sin>
py: Call <MxFunc: sin>

ans =

    0.7071

Ref count: 1
Ref count: 0
%}
