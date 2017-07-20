x = reshape(1:4, [2, 2]);

t = SubsrefDispatch(x);

t(1, 2)
t.Value(1, 2)
t.Value(1, :)
t.Value(:)

t.doStuff(5)

t.doStuff(5).out
