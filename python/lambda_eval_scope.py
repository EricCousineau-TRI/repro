# https://github.com/pybind/pybind11/issues/1742

a = 123

def make_f():
    a = 42
    print(locals())  # outputs {'a': 42}
    fe = eval('lambda: a')
    fc = lambda: a
    return fe, fc

fe, fc = make_f()
print(fe())   # outputs 123
print(fc())   # outputs 42

locals_ = {'a': 420}
print(eval('a', {}, locals_))  # outputs 420

eval('lambda: a', {}, locals_)()  # ERROR: 'a' is not defined
exec('f = lambda: a', {}, locals_); print(locals_["f"]())  # Same error
