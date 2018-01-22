# pip install wrapt

# Prototype for enabling a semi-transparent layer of `const`-honoring Pythonic
# stuff.
# N.B. There should *always* be an option for disabling this for performance
# reasons!

import inspect
from wrapt import ObjectProxy
from types import MethodType

class ConstError(RuntimeError):
    pass

def _is_literal(cls):
    if cls in [int, float, str, unicode, tuple]:
        return True
    else:
        return False

def _is_method_of(func, obj):
    return inspect.ismethod(func) and func.im_self is obj

def _rebind_method(bound, new_self):
    # https://stackoverflow.com/a/14574713/7829525
    return MethodType(bound.__func__, new_self, bound.im_class)

class _ConstClassMeta(object):
    def __init__(self, cls, owned=None, mutable=None):
        self._cls = cls
        if owned is None:
            owned = set()
        if mutable is None:
            mutable = set()
        self._owned = set(owned)  # set of strings
        self.mutable = set(mutable)  # set of strings
        # Add any decorated mutable methods.
        methods = inspect.getmembers(cls, predicate=inspect.ismethod)
        for name, method in methods:
            if getattr(method, '_is_mutable_method', False):
                mutable.add(name)
        # Recursion / inheritance is handled by `_ConstClassMetaMap`, to handle
        # caching.

    def update(self, parent):
        self._owned.update(parent._owned)
        self.mutable.update(parent.mutable)

    def is_owned(self, name):
        return name in self._owned

    def is_mutable(self, name):
        # Limitation: This would not handle overloads.
        # (e.g. `const T& get() const` and `T& get()`.
        return name in self.mutable

class _ConstClassMetaMap(object):
    def __init__(self):
        self._meta_map = {}

    def emplace(self, cls, owned=None, mutable=None):
        meta = _ConstClassMeta(cls, owned, mutable)
        return self._add(cls, meta)

    def _add(self, cls, meta):
        assert cls not in self._meta_map
        # Recurse through bases for complete information.
        for base in cls.__bases__:
            meta.update(self.get(base))
        self._meta_map[cls] = meta
        return meta

    def get(self, cls):
        # Assume class has not changed if it's already registered.
        meta = self._meta_map.get(cls, None)
        if meta:
            return meta
        else:
            # Construct default.
            return self._add(cls, _ConstClassMeta(cls))

_const_metas = _ConstClassMetaMap()
_emplace = _const_metas.emplace
# Register common mutators.
# N.B. These methods actually have to be overridde in `Const`, since neither
# `__getattr__` nor `__getattribute__` will capture them.
_emplace(object, mutable={
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
    })
_emplace(list, mutable={
    'append', 'clear', 'insert', 'extend', 'insert', 'pop',
    'remove', 'sort'})
_emplace(dict, mutable={'clear', 'setdefault'})

class _Const(ObjectProxy):
    def __init__(self, wrapped):
        ObjectProxy.__init__(self, wrapped)

    def __getattr__(self, name):
        wrapped = object.__getattribute__(self, '__wrapped__')
        meta = _const_metas.get(type(wrapped))
        value = getattr(wrapped, name)
        if meta.is_mutable(name):
            # If explicitly mutable, raise an error.
            # (For complex situations.)
            _raise_mutable_method_error(self, name)
        elif _is_method_of(value, wrapped):
            # Propagate const-ness into method (replace `im_self`) to catch
            # basic violations.
            return _rebind_method(value, self)
        elif meta.is_owned(name):
            # References (pointer-like things) should not be const, but
            # internal values should.
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

# Automatically rewrite any mutable methods for `object` to always deny access.
for name in _const_metas.get(object).mutable:
    def _new_scope(name):
        def _no_access(self, *args, **kwargs):
            _raise_mutable_method_error(self, name)
        _no_access.__name__ = name
        setattr(_Const, name, _no_access)
    _new_scope(name)


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
    if obj is not None:
        if _is_literal(type(obj)):
            # Literals / immutable types have no references.
            # No need to waste a proxy.
            return obj
        else:
            return _Const(obj)

def to_mutable(obj, force=False):
    # Forced conversion.
    if not force and is_const(obj):
        raise ConstError(
            "Cannot cast const {} to mutable instance".format(type_ex(self)))
    if isinstance(obj, _Const):
        return obj.__wrapped__
    elif isinstance(obj, ConstIter):
        return obj._iter
    else:
        return obj

def is_const(obj):
    if isinstance(obj, _Const) or isinstance(obj, ConstIter):
        return True
    else:
        return False

def type_ex(obj):
    if isinstance(obj, _Const):
        return type(obj.__wrapped__)
    elif isinstance(obj, ConstIter):
        return type(obj._iter)
    else:
        return type(obj)

def _raise_mutable_method_error(self, name):
    raise ConstError(
        ("'{}' is a mutable method that cannot be called on a " +
        "const {}.").format(name, type_ex(self)))

def mutable(func):
    func._is_mutable_method = True
    return func

def add_const_meta(cls, owned=None, mutable=None):
    _const_metas.emplace(cls, owned, mutable)
    return cls

def const_meta(owned=None, mutable=None):
    return lambda cls: add_const_meta(cls, owned, mutable)

@const_meta(owned = ['_map'])
class Check(object):
    def __init__(self, value):
        self._value = value
        self._map = {10: 0}

    def get_value(self):
        return self._value

    def print_self(self):
        print(type(self), self)

    def set_value(self, value):
        self._value = value

    def get_map(self, key):
        return self._map[key]

    def set_map(self, key, value):
        self._map[key] = value

    def do_something(self, stuff):
        print("{}: {}".format(stuff, self._value))
        # self.set_value(10)  # Raises error.

    @mutable
    def mutate(self): pass

    def __setitem__(self, key, value):
        self._map[key] = value

    value = property(get_value, set_value)

@const_meta(owned = ['_my_value'])
class SubCheck(Check):
    def __init__(self):
        Check.__init__(self, 100)
        self._my_value = []

    def extra(self):
        self.set_map(10, 10000)

    def more(self):
        # self._my_value.append(10)  # Raises error.
        return self._my_value

c = Check(10)
c_const = to_const(c)
print(c_const.value)
# c_const.value = 100  # Raises error.
# c_const.set_value(100)  # Raises error.
c_const.do_something("yo")
print(c_const == c_const)
print(c_const.get_map(10))
# print(c_const.set_map(10, 100))  # Raises error.
# c_const[10] = 10  # Raises error.

c.value = 100
print(c_const.value)

print(c_const)
print(c_const.__dict__)
print(type(c_const.__dict__))
print(type_ex(c_const.__dict__))
# c_const.__dict__['value'] = 200

s = SubCheck()
s_const = to_const(s)

c.print_self()
c_const.print_self()
s_const.print_self()
# s_const.extra()  # Raises error.
# s_const.mutate()  # Raises error.
x = s_const.more()
print(x)
print(type(x))
# x[:] = []  # Raises error.

obj = to_const([1, 2, [10, 10]])
print(is_const(obj))
print(is_const([]))
# obj[1] = 3  # Raises error.
# to_mutable(obj)[1] = 10  # Raises error.
to_mutable(obj, force=True)[1] = 10
print(obj)

for i in obj:
    print(i)
    if isinstance(i, list):
        print("woo")
        # i[0] = 10  # Raises error.
