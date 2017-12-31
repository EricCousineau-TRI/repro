#!/usr/bin/env python

# Template definitions.
from pymodule.tpl.cpp_tpl_types import type_registry

_PARAM_DEFAULT = 'first_registered'

class Template(object):
    def __init__(self, param_default=_PARAM_DEFAULT):
        if isinstance(param_default, tuple) or isinstance(param_default, list):
            param_default = self.param_canonical(param_default)
        self._param_default = param_default
        self.param_list = []
        self._instantiation_map = {}

    def param_canonical(self, param):
        if not isinstance(param, tuple):
            param = tuple(param)
        return type_registry.GetPyTypesCanonical(param)

    def __getitem__(self, param):
        """ Gets concrete class associate with the given arguments.
        If called with [[]], then returns the default instantiation. """
        if isinstance(param, tuple):
            return self.get_instantiation(param)
        else:
            # Scalar type.
            return self.get_instantiation((param,))

    def get_instantiation(self, param=[[]]):
        if len(param) == 1 and param[0] == []:
            assert self._param_default is not None
            param = self._param_default
        param = self.param_canonical(param)
        return self._instantiation_map[param]

    def add_instantiation(self, param, instantiation):
        """ Adds instantiation. """
        # Ensure that we do not already have this tuple.
        param = self.param_canonical(param)
        assert param not in self._instantiation_map, "Instantiation already registered"
        # Add it.
        self.param_list.append(param)
        self._instantiation_map[param] = instantiation
        if self._param_default == _PARAM_DEFAULT:
            self._param_default = param
        return param


class TemplateClass(Template):
    def __init__(self, name, parent=None, param_default=_PARAM_DEFAULT):
        Template.__init__(self, param_default=param_default)
        self.name = name
        self.parent = parent

    def add_instantiation(self, param, cls):
        """ Adds instantiation. """
        # Do not double-register existing instantiation.
        if is_tpl_cls(cls):
            if self.parent and cls._tpl == self.parent:
                pass
            else:
                # Do not permit any existing template.
                raise RuntimeError("Class already has template associated with it")
        # Nominal behavior.
        param = Template.add_instantiation(self, param, cls)
        # Update class.
        cls._tpl = self
        param_str = map(type_registry.GetCppName, param)
        tpl_name = '{}[{}]'.format(self.name, ', '.join(param_str))
        cls.__name__ = tpl_name

    def add_classes_with_factory(self, cls_factory, param_list=None):
        if param_list is None:
            assert self.parent is not None
            param_list = self.parent.param_list
        else:
            param_list = map(self.param_canonical, param_list)
        for param in param_list:
            cls = cls_factory(param)
            self.add_instantiation(param, cls)


def is_tpl_cls(cls):
    return hasattr(cls, '_tpl') and isinstance(cls._tpl, Template)


def is_tpl_of(cls, tpl):
    return is_tpl_cls(cls) and cls._tpl == tpl


class TemplateMethod(Template):
    def __init__(self, name, param_default=_PARAM_DEFAULT):
        Template.__init__(self, param_default=param_default)

    def bind(self, obj):
        return _TemplateMethodBound(self, obj)


class _TemplateMethodBound(object):
    def __init__(self, tpl, obj):
        self._tpl = tpl
        self._obj = obj

    def __getitem__(self, param):
        # TODO: Figure out actual binding.
        unbound = self._tpl[param]
        obj = self._obj
        def bound(*args, **kwargs):
            return unbound(obj, *args, **kwargs)
        return bound
