#!/usr/bin/env python

# Template definitions.
from pymodule.tpl.cpp_tpl_types import type_registry

def _tpl_name(name, param):
    param_str = map(type_registry.GetCppName, param)
    return '{}[{}]'.format(name, ', '.join(param_str))


class Template(object):
    def __init__(self, name, param_default=None):
        self.name = name
        self._param_default = tuple(param_default)
        self._cls_map = {}

    def __getitem__(self, param):
        """ Gets concrete class associate with the given arguments.

        If called with [[]], then returns the default instantiation. """
        if isinstance(param, tuple):
            return self.get_class(param)
        else:
            # Scalar type.
            return self.get_class((param,))

    def get_class(self, param=[[]]):
        if len(param) == 1 and param[0] == []:
            assert self._param_default is not None
            param = self._param_default
        cls = self._cls_map[param]
        return cls

    def add_class(self, param, cls):
        """ Adds instantiation. """
        # Do not double-register existing instantiation.
        if is_tpl_cls(cls):
            self._check_tpl_cls(cls)
        param = tuple(param)
        # Ensure that we do not already have this tuple.
        assert param not in self._cls_map, "Param tuple already registered"
        # Add it.
        self._cls_map[param] = cls
        if self._param_default == 'first_registered':
            self._param_default = param
        # Update class.
        cls._tpl = self
        cls.__name__ = _tpl_name(self.name, param)

    def _check_tpl_cls(self, cls):
        # Do not permit any existing template.
        raise RuntimeError("Class already has template associated with it")


class ChildTemplate(Template):
    def __init__(self, name, parent, **kwargs):
        Template.__init__(self, name, parent._param_default, **kwargs)
        self._parent = parent

    def add_classes_with_factory(self, cls_factory, param_list=None):
        if param_list is None:
            param_list = self._parent._cls_map.keys()
        for param in param_list:
            cls = cls_factory(*param)
            # Sanity check.
            base_cls = self._parent[param]
            assert issubclass(cls, base_cls)
            self.add_class(param, cls)

    def _check_tpl_cls(self, cls):
        # Permit inherited tpl class ONLY.
        if cls._tpl == self._parent:
            pass
        else:
            # Raise the original error.
            raise RuntimeError("Class already has template associated with it (and is not the parent template)")


def is_tpl_cls(cls):
    return hasattr(cls, '_tpl') and isinstance(cls._tpl, Template)


def is_tpl_of(cls, tpl):
    return is_tpl_cls(cls) and cls._tpl == tpl
