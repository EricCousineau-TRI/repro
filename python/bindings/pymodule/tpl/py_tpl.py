#!/usr/bin/env python

# Template definitions.
from pymodule.tpl.cpp_tpl_types import type_registry


class Template(object):
    def __init__(self, name, parent=None, param_default='first_registered'):
        self.name = name
        if isinstance(param_default, tuple) or isinstance(param_default, list):
            param_default = self.param_canonical(param_default)
        self._param_default = param_default
        self._param_list = []
        self._parent = parent
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
        # Canonicalize.
        param = self.param_canonical(param)
        cls = self._cls_map[param]
        return cls

    def param_canonical(self, param):
        return type_registry.GetPyTypesCanonical(param)

    def add_class(self, param, cls):
        """ Adds instantiation. """
        # Do not double-register existing instantiation.
        if is_tpl_cls(cls):
            if self._parent and cls._tpl == self._parent:
                pass
            else:
                # Do not permit any existing template.
                raise RuntimeError("Class already has template associated with it")
        # Ensure that we do not already have this tuple.
        param = self.param_canonical(param)
        assert param not in self._cls_map, "Param tuple already registered"
        # Add it.
        self._param_list.append(param)
        self._cls_map[param] = cls
        if self._param_default == 'first_registered':
            self._param_default = param
        # Update class.
        cls._tpl = self
        param_str = map(type_registry.GetCppName, param)
        tpl_name = '{}[{}]'.format(self.name, ', '.join(param_str))
        cls.__name__ = tpl_name

    def add_classes_with_factory(self, cls_factory, param_list=None):
        if param_list is None:
            assert self._parent is not None
            param_list = self._parent._param_list
        else:
            param_list = map(self.param_canonical, param_list)
        for param in param_list:
            cls = cls_factory(param)
            self.add_class(param, cls)


def is_tpl_cls(cls):
    return hasattr(cls, '_tpl') and isinstance(cls._tpl, Template)


def is_tpl_of(cls, tpl):
    return is_tpl_cls(cls) and cls._tpl == tpl
