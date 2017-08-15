% Hopeful example

% Create instance, strong reference, and weak reference.
x = Obj([1, 2, 3]);
x_strong = x;
x_weak = weakref(x);  % Is this possible???

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
assert(isempty(x_weak.get()));
