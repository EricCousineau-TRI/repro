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

%%
% finv = @(A, b) A \ b;
e = py_simple.call_check(@mldivide, 2 * eye(2), [2, 4]')

%%
% TODO: See if there is a way to support this?
py_simple.call_check(@fieldnames, struct('test', 1))

%%
%{
MxFunc: @(x)x/pi>

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
%}
