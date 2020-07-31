"""
@file
Enables a semi-transparent layer of C++ const-honoring Python proxies.

Requires: `pip install wrapt`
"""

import inspect
from wrapt import ObjectProxy
from types import MethodType

# N.B. There should *always* be an option for disabling this for performance
# reasons!

# TODO(eric.cousineau): How to handle levels of recursion?
# e.g. should __getitem__ always return const-proxied values?


class ConstError(TypeError):
    pass


class _ConstClassMeta(object):
    # Provides metadata for a given class.
    def __init__(self, cls, owned_properties=None, mutable_methods=None):
        self.cls = cls
        self.owned_properties = set(owned_properties or set())  # set of strings
        self.mutable_methods = set(mutable_methods or set())  # set of strings
        # Add any decorated mutable methods.
        methods = inspect.getmembers(cls, predicate=inspect.ismethod)
        for name, method in methods:
            # TODO(eric.cousineau): Warn if there is a mix of mutable and
            # immutable methods with the same name.
            if getattr(method, '_is_mutable_method', False):
                self.mutable_methods.add(name)
        # Handle inheritance.
        for base_cls in self.cls.__bases__:
            base_meta = _const_metas.get(base_cls)  # Minor cyclic dependency.
            self.owned_properties.update(base_meta.owned_properties)
            self.mutable_methods.update(base_meta.mutable_methods)

    def is_owned_property(self, name):
        return name in self.owned_properties

    def is_mutable_method(self, name):
        # Limitation: This would not handle overloads.
        # (e.g. `const T& get() const` and `T& get()`.
        return name in self.mutable_methods


class _ConstClassMetaMap(object):
    # Provides mapping from class to metadata.
    def __init__(self):
        self._meta_map = {}

    def emplace(self, cls, owned_properties=None, mutable_methods=None):
        meta = _ConstClassMeta(cls, owned_properties, mutable_methods)
        return self._add(cls, meta)

    def _add(self, cls, meta):
        assert cls not in self._meta_map
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
_emplace(object, mutable_methods={
    "__setattr__",
    "__delattr__",
    "__setitem__",
    "__delitem__",
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
_emplace(list, mutable_methods={
    'append', 'clear', 'insert', 'extend', 'insert', 'pop',
    'remove', 'sort'})
_emplace(dict, mutable_methods={'clear', 'setdefault'})


class _Const(ObjectProxy):
    def __init__(self, wrapped):
        ObjectProxy.__init__(self, wrapped)

    def __getattr__(self, name):
        # Intercepts access to mutable methods, general methods, and owned
        # properties.
        wrapped = object.__getattribute__(self, '__wrapped__')
        meta = _const_metas.get(type(wrapped))
        value = getattr(wrapped, name)
        if meta.is_mutable_method(name):
            # If decorated as a mutable method, raise an error. Do not allow
            # access to the bound method, because the only way this method
            # *should* become usable is to rebind it.
            _raise_mutable_method_error(self, name)
        elif _is_method_of(value, wrapped):
            # Rebind method to const `self` recursively catch basic violations.
            return _rebind_method(value, self)
        elif meta.is_owned_property(name):
            # References (pointer-like things) should not be const, but
            # internal values should.
            return to_const(value)
        else:
            return value

    @property
    def __dict__(self):
        return to_const(self.__wrapped__.__dict__)

    def __iter__(self):
        # TODO(eric.cousineau): This assumes that iterating will yield owned
        # items. Add option to suppress this?
        return _ConstIter(self.__wrapped__)

    def __str__(self):
        out = self.__wrapped__.__str__()
        if (len(out) >= 2 and len(out) < 200 and
                out[0] == '<' and out[-1] == '>'):
            return '<const ' + out[1:]
        else:
            return out


# Automatically rewrite any mutable methods for `object` to deny access upon
# calling.
for name in _const_metas.get(object).mutable_methods:
    def _capture(name):
        def _no_mutable(self, *args, **kwargs):
            _raise_mutable_method_error(self, name)
        _no_mutable.__name__ = name
        setattr(_Const, name, _no_mutable)
    _capture(name)


class _ConstIter(object):
    # Provides a const-proxying iterator.
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.__iter__()

    def __iter__(self):
        self._wrapped_iter = iter(self._wrapped)
        return self

    def next(self):
        n = next(self._wrapped_iter)
        return to_const(n)


def _is_immutable(obj):
    # Detects if a type is a immutable (literal) type.
    literal_types = [int, float, str, unicode, tuple, type(None)]
    if type(obj) in literal_types:
        return True
    else:
        return False


def _is_method_of(func, obj):
    # Detects if `func` is a function bound to a given instance `obj`.
    return inspect.ismethod(func) and func.im_self is obj


def _rebind_method(bound, new_self):
    # Rebinds `bound.im_self` to `new_self`.
    # https://stackoverflow.com/a/14574713/7829525
    return MethodType(bound.__func__, new_self, bound.im_class)


def to_const(obj):
    """Converts an object to a const proxy. Does not proxy immutable / literal
    types. """
    if _is_immutable(obj):
        return obj
    else:
        return _Const(obj)


def to_mutable(obj, force=False):
    """Converts to a mutable (non-const proxied) object.
    If `force` is False, will throw an error if `obj` is const. """
    if not force and is_const(obj):
        raise ConstError(
            "Cannot cast const {} to mutable instance"
            .format(type_extract(self)))
    if isinstance(obj, _Const):
        return obj.__wrapped__
    elif isinstance(obj, _ConstIter):
        return obj._iter
    else:
        return obj


def is_const(obj):
    """Determines if `obj` is const-proxied. """
    if isinstance(obj, _Const) or isinstance(obj, _ConstIter):
        return True
    else:
        return False


def is_const_or_immutable(obj):
    return is_const(obj) or _is_immutable(obj)


def type_extract(obj):
    """Extracts type from an object if it's const-proxied; otherwise returns
    direct type. """
    if isinstance(obj, _Const):
        return type(obj.__wrapped__)
    elif isinstance(obj, _ConstIter):
        return type(obj._iter)
    else:
        return type(obj)


def _raise_mutable_method_error(obj, name):
    raise ConstError(
        ("'{}' is a mutable method that cannot be called on a " +
        "const {}.").format(name, type_extract(obj)))


def mutable_method(func):
    """Decorates a function as mutable. """
    func._is_mutable_method = True
    return func


def _add_const_meta(cls, owned_properties=None, mutable_methods=None):
    # Adds const-proxy metadata to a class.
    _const_metas.emplace(cls, owned_properties, mutable_methods)
    return cls


def const_meta(owned_properties=None, mutable_methods=None):
    """Decorates a class with const-proxy metadata. """
    return lambda cls: _add_const_meta(cls, owned_properties, mutable_methods)
