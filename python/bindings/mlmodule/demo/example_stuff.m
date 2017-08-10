% Call directly from Python
ex = pyimport_proxy('example_mod');
% Reload it
py.reload(ex.py);

%%
ex.example(1);
ex.example(eye(5));
