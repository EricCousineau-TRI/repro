e = Erasure();

values = {1, struct('hello', 1)};

i1 = e.store(1);
i2 = e.store(struct('hello', 1));

struct(e)
e.retrieve(i2)
e.decrementReference(i2)
struct(e)
e.retrieve(i1)
e.decrementReference(i1)

assert(e.count() == 0);

%% Should error out
e.retrieve(i2)
