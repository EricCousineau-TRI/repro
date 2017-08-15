% Shortened version of `ml_example`
x = Obj([1, 2, 3]);
x_weak = weakref(x, true);
clear x
if ~isempty(x_weak.get())
    warning('weak ref (a) did not work\n');
end
clear x_weak

%%
x = Obj([4, 5, 6]);
x_weak = weakref(x, false);
clear x
if ~isempty(x_weak.get())
    warning('weak ref (b) did not work\n');
end
clear x_weak
