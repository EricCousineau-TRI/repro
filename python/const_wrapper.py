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
    "__dict__",
]

wrap_functions = [
    "__iter__",
]

class ConstError(RuntimeError):
    pass

class Const(ObjectProxy):
    def __init__(self, wrapped):
        ObjectProxy.__init__(self, wrapped)

    def _is_mutable(self, name):
        wrapped = self.__wrapped__
        return False

    @property
    def __dict__(self):
        wrapped = self.__wrapped__
        return Const(wrapped.__dict__)

    def const_cast(self):
        return self.__wrapped__

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

# for f in wrap_attr:
#     old_property = do_get(Const, f)
#     print(old_property)
#     print(type(old_property))
#     assert old_property.fset is None
#     assert old_property.fdel is None
#     fget = old_property.fget
#     def _fget_wrap(self):
#         return Const(fget(self))
#     new_property = property(_fget_wrap)
#     do_set(Const, f, new_property)

for f in wrap_functions:
    func = do_get(ObjectProxy, f)
    def _func_wrap(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        # TODO: Check if literal type?
        if out is not None:
            return Const(out)
    do_set(Const, f, _func_wrap)


class Check(object):
    def __init__(self, value):
        self._value = value
        self._map = dict()

    def _get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = value

    def do_something(self, stuff):
        print("{}: {}".format(stuff, self._value))

    def __setitem__(self, key, value):
        self._map[key] = value

    value = property(_get_value, _set_value)


c = Check(10)
c_const = Const(c)

print(c_const.value)
# c_const.value = 100
c_const.do_something("yo")
print(c_const == c_const)
# c_const[10] = 10

c.value = 100
print(c_const.value)

print(c_const.__dict__)
print(type(c_const.__dict__))
# c_const.__dict__['value'] = 200

obj = Const([1, 2, 3])
# obj[1] = 3
obj.const_cast()[1] = 10
print(obj)
