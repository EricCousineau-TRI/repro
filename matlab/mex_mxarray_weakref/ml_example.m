% Hopeful example
% Should have been a working but simplified version of:
% https://www.mathworks.com/matlabcentral/answers/287708-some-matlab-versions-crash-when-using-listener

% Create instance, strong reference, and weak reference.
x = Obj([1, 2, 3]);
x_strong = x;
x_weak = weakref(x, true);  % Is this possible???
% x_weak = weakref(x, false);  % Is this possible???

% Delete original reference. Strong reference keeps alive.
fprintf('clear x\n');
clear x
fprintf('strong ref: \n'); disp(x_strong);
fprintf('weak ref: \n'); disp(x_weak.get());

% Delete strong reference. Weak reference does not keep alive.
fprintf('\nclear x_strong\n');
clear x_strong
fprintf('weak ref: \n'); disp(x_weak.get());
% How to make this work?
if ~isempty(x_weak.get())
    warning('weak reference still alive\n');
end

%%
fprintf('\nclear x_weak\n');
clear x_weak
