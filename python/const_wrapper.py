# pip install wrapt

from wrapt import ObjectProxy

mutable_functions = [
    "__setattr__",
    "__delattr__",
    "__setitem__",
    "__setslice__",
    "__iadd__",
    "__isub__",
    "__imul__",
    "__idiv__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
    "__enter__",
    "__exit__",
]

# Functions / properties whose return value should be wrapped as const.
wrap_attr = [
    # "__dict__",
]

wrap_functions = []

class ConstError(RuntimeError):
    pass

class Const(ObjectProxy):
    def __init__(self, wrapped):
        ObjectProxy.__init__(self, wrapped)

    def __iter__(self):
        return ConstIter(self.__wrapped__)

    def __str__(self):
        out = self.__wrapped__.__str__()
        if len(out) >= 2 and len(out) < 200 and out[0] == '<' and out[-1] == '>':
            return '<const ' + out[1:]
        else:
            return out

    @property
    def __dict__(self):
        return to_const(self.__wrapped__.__dict__)

class ConstIter(object):
    def __init__(self, obj):
        self._obj = obj
        self.__iter__()

    def __iter__(self):
        self._iter = iter(self._obj)
        return self

    def next(self):
        n = next(self._iter)
        return to_const(n)

def to_const(obj):
    # TODO: Check if literal type?
    if obj is not None:
        return Const(obj)

def const_cast(obj):
    if isinstance(obj, Const):
        return obj.__wrapped__
    else:
        return obj

def is_const(obj):
    if isinstance(obj, Const) or isinstance(obj, ConstIter):
        return True
    else:
        return False

def type_ex(obj):
    if isinstance(obj, Const):
        return type(obj.__wrapped__)
    elif isinstance(obj, ConstIter):
        return type(obj._iter)
    else:
        return type(obj)

do_get = object.__getattribute__
do_set = setattr #object.__setattr__

for f in mutable_functions:
    def _new_scope(f):
        message = "'{}' is a mutable function".format(f)
        def _no_access(*args, **kwargs):
            raise ConstError(message)
        _no_access.__name__ = f
        do_set(Const, f, _no_access)
    _new_scope(f)

for f in wrap_attr:
    def _new_scope(f):
        def wrap(self):
            print("wrap: {}".format(f))
            wrapped = self.__wrapped__
            return to_const(do_get(wrapped, f))
        wrap.__name__ = f
        do_set(Const, f, property(wrap))

for f in wrap_functions:
    def _new_scope(f):
        func = do_get(ObjectProxy, f)
        def wrap(self, *args, **kwargs):
            return to_const(func(self, *args, **kwargs))
        wrap.__name__ = f
        do_set(Const, f, wrap)
    _new_scope(f)

def mutable(f):
    f._is_mutable = True
    return f

class Check(object):
    def __init__(self, value):
        self._value = value
        self._map = dict()

    def get_value(self):
        return self._value

    def print_self(self):
        print(type(self), self)

    @mutable
    def _other_method(self):
        self._value

    @mutable
    def set_value(self, value):
        self._value = value

    def do_something(self, stuff):
        print("{}: {}".format(stuff, self._value))

    def __setitem__(self, key, value):
        self._map[key] = value

    value = property(get_value, set_value)

print(Const.__iter__)

c = Check(10)
c_const = to_const(c)

print(c_const.value)
# c_const.value = 100
c_const.set_value(100)
c_const.do_something("yo")
print(c_const == c_const)
# c_const[10] = 10

c.value = 100
print(c_const.value)

print(c_const)
print(c_const.__dict__)
print(type(c_const.__dict__))
print(type_ex(c_const.__dict__))
# c_const.__dict__['value'] = 200

c.print_self()
c_const.print_self()
Check.print_self(c_const)

obj = to_const([1, 2, [10, 10]])
print(is_const(obj))
print(is_const([]))
# obj[1] = 3
const_cast(obj)[1] = 10
print(obj)

for i in obj:
    print(i)
    if isinstance(i, list):
        print("woo")
        # i[0] = 10
