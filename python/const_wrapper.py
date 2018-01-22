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

from types import MethodType

def _is_method_of(func, obj):
    return hasattr(func, 'im_class') and func.im_self is obj

def _rebind_method(bound, new_self):
    # https://stackoverflow.com/a/14574713/7829525
    return MethodType(bound.__func__, new_self, bound.im_class)

class _ConstRegistry(object):
    def __init__(self):
        self._owned_names = {}

    def _get_names(self, cls):
        # Assume class has not changed if it's already registered.
        names = self._owned_names.get(cls, None)
        if names is not None:
            return names
        names = set(getattr(cls, '_owned_objects', []))
        for base in cls.__bases__:
            names.update(self._get_names(base))
        self._owned_names[cls] = names
        return names

    def should_wrap(self, wrapped, name):
        cls = type(wrapped)
        return name in self._get_names(cls)

_const_registry = _ConstRegistry()

class Const(ObjectProxy):
    def __init__(self, wrapped):
        ObjectProxy.__init__(self, wrapped)

    def __getattr__(self, name):
        wrapped = object.__getattribute__(self, '__wrapped__')
        value = getattr(wrapped, name)
        # NOTE: If a callback or something has a reference to `self`, then
        # this will fall apart. Oh well.
        if _is_method_of(value, wrapped):
            # If explicitly mutable, raise an error.
            # (For complex situations.)
            if getattr(value, '_is_mutable_method', False):
                _raise_error(self, name)
            # Propagate const-ness into method (replace `im_self`) to catch
            # basic violations.
            return _rebind_method(value, self)
        else:
            # References (pointer-like things) should not be const, but
            # internal values should...
            if _const_registry.should_wrap(wrapped, name):
                return to_const(value)
            else:
                return value

    @property
    def __dict__(self):
        return to_const(self.__wrapped__.__dict__)

    def __iter__(self):
        return ConstIter(self.__wrapped__)

    def __str__(self):
        out = self.__wrapped__.__str__()
        if (len(out) >= 2 and len(out) < 200 and
                out[0] == '<' and out[-1] == '>'):
            return '<const ' + out[1:]
        else:
            return out

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

def _raise_error(self, name):
    raise ConstError(
        ("'{}' is a mutable method that cannot be called on a " +
        "const {}.").format(name, type_ex(self)))

for f in mutable_functions:
    def _new_scope(f):
        def _no_access(self, *args, **kwargs):
            _raise_error(self, f)
        _no_access.__name__ = f
        do_set(Const, f, _no_access)
    _new_scope(f)

def mutable(f):
    f._is_mutable_method = True
    return f



class Check(object):
    _owned_objects = ['_map']

    def __init__(self, value):
        self._value = value
        self._map = {10: 0}

    def get_value(self):
        return self._value

    def print_self(self):
        print(type(self), self)

    def _other_method(self):
        self._value

    def set_value(self, value):
        self._value = value

    def get_map(self, key):
        return self._map[key]

    def set_map(self, key, value):
        self._map[key] = value

    def do_something(self, stuff):
        print("{}: {}".format(stuff, self._value))
        # self.set_value(10)  # Causes error.

    @mutable
    def mutate(self): pass

    def __setitem__(self, key, value):
        self._map[key] = value

    value = property(get_value, set_value)

class SubCheck(Check):
    def __init__(self):
        Check.__init__(self, 100)

    def extra(self):
        self.set_map(10, 10000)

print(Const.__iter__)

c = Check(10)
c_const = to_const(c)

print(c_const.value)
# c_const.value = 100
# c_const.set_value(100)
c_const.do_something("yo")
print(c_const == c_const)
print(c_const.get_map(10))
# print(c_const.set_map(10, 100))
# c_const[10] = 10

c.value = 100
print(c_const.value)

print(c_const)
print(c_const.__dict__)
print(type(c_const.__dict__))
print(type_ex(c_const.__dict__))
# c_const.__dict__['value'] = 200

c.print_self()
print(c.print_self.im_class)
c_const.print_self()
Check.print_self(c_const)

s = SubCheck()
s_const = to_const(s)
s_const.print_self()
# s_const.extra()
s_const.mutate()

# obj = to_const([1, 2, [10, 10]])
# print(is_const(obj))
# print(is_const([]))
# # obj[1] = 3
# const_cast(obj)[1] = 10
# print(obj)

# for i in obj:
#     print(i)
#     if isinstance(i, list):
#         print("woo")
#         # i[0] = 10
