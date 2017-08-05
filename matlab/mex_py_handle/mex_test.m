%%
make;
clear classes; %#ok<CLCLS>
MexPyProxy.init();

%%
% mex_py_proxy('py_so_reload');
x = MexPyProxy.test_call(@sin, pi / 4)

%%
x = MexPyProxy.test_call(@bad_func, pi / 4)

% mex_py_proxy('mx_feval_py_raw');

%%

% %%
% values = {1, eye(20, 20)};  % Fails
% % Be wary of passing temporaries! May be subject to ML gargabe collection.
% for i = 1:length(values)
%     value = values{i};
% %     value = 1;
%     mx_raw = mex_py_proxy('mx_to_mx_raw', value);
%     mx_value = mex_py_proxy('mx_raw_to_mx', mx_raw, value);
%     if ~isequal(value, mx_value)
%         value
%         mx_value
%         warning('Not equal');
%     end
% end
