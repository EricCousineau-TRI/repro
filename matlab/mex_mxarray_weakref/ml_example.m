% Hopeful example

% Create instance, strong reference, and weak reference.
x = Obj([1, 2, 3]);
x_strong = x;
% x_weak = weakref(x, false);  % This does not call the "Destroyed" callback.
x_weak = weakref(x, false);  % This DOES call the "Destroyed" callback.
% However, if the reference is stored, then `weakref` is not really a weak
% reference.

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
    warning('Did not work\n');
end

%%
clear x_weak
