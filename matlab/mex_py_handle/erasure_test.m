e = Erasure();

values = {1, struct('hello', 1)};

i1 = e.push(1);
i2 = e.push(struct('hello', 1));

struct(e)
e.pop(i2)
struct(e)
e.pop(i1)

% Should error out
e.pop(i2)

%%
e = Erasure();

values = {1, struct('hello', 1)};

i1 = MexPyProxy.mx_to_mx_raw(1);
i2 = MexPyProxy.mx_to_mx_raw(struct('hello', 1));

MexPyProxy.mx_raw_to_mx(i2)
MexPyProxy.mx_raw_to_mx(i1)
