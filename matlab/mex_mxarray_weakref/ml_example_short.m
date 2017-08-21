x = Instance([4, 5, 6]);

% Store weak reference.
x_weak = weakref(x);

% % This does work as you mentioned. However, I would like to know why
% % "clear" does not work in the simplified "weakref".
% delete(x)

% This does not call the callback, which implies that there *is* a coupling
% between the listener and the source's lifetime.
clear x

% Programmatic check - would expect the source to empty, or something.
x_ref = x_weak.get();
if ~isempty(x_ref)
    % Check if it's the same value.
    assert(isequal(x_ref.Value, [4, 5, 6]));
    warning('weak ref (b) did not work\n');
else
    fprintf('Worked-ish\n');
end
clear x_ref

% Callback is still not called, even though .get() worked (and the instance
% is still refernenced in the listener).
% Did it get dropped without getting called?
clear x_weak

% clear all