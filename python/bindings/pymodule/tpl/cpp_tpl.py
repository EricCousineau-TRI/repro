#!/usr/bin/env python

import inspect
import types

# Template definitions.
from pymodule.tpl.cpp_tpl_types import type_registry

PARAM_DEFAULT = ([],)
_PARAM_DEFAULT_DEFAULT = 'first_registered'


class Template(object):
    def __init__(self, name, param_default=_PARAM_DEFAULT_DEFAULT, module_name=None):
        self.name = name
        if isinstance(param_default, tuple) or isinstance(param_default, list):
            param_default = self.param_canonical(param_default)
        self._param_default = param_default
        self.param_list = []
        self._instantiation_map = {}
        if module_name is None:
            module_name = inspect.getmodule(inspect.stack()[1][0]).__name__
        self._module_name = module_name

    def _param_resolve(self, param):
        # Resolve from default argument to canonical parameters.
        if not isinstance(param, tuple):
            param = tuple(param)
        if param == PARAM_DEFAULT:
            self._param_default is not None
            param = self._param_default
        return self.param_canonical(param)

    def param_canonical(self, param):
        """Gets canonical parameter pack that makes it simple to mesh with
        C++ types. """
        return type_registry.GetPyTypesCanonical(param)

    def __getitem__(self, param):
        """ Gets concrete class associate with the given arguments.
        If called with [[]], then returns the default instantiation. """
        if not isinstance(param, tuple) or isinstance(param, list):
            # Handle scalar case.
            param = (param,)
        return self.get_instantiation(param)

    def get_instantiation(self, param=PARAM_DEFAULT):
        param = self._param_resolve(param)
        return self._instantiation_map[param]

    def _get_instantiation_name(self, param):
        param_str = map(type_registry.GetName, param)
        return '{}[{}]'.format(self.name, ', '.join(param_str))

    def add_instantiation(self, param, instantiation):
        """ Adds instantiation. """
        # Ensure that we do not already have this tuple.
        param = self.param_canonical(param)
        assert param not in self._instantiation_map, "Instantiation already registered"
        # Add it.
        self.param_list.append(param)
        self._instantiation_map[param] = instantiation
        if self._param_default == _PARAM_DEFAULT_DEFAULT:
            self._param_default = param
        return param

    def add_instantiations(
            self, instantiation_func, param_list=None):
        assert param_list is not None
        for param in param_list:
            self.add_instantiation(param, instantiation_func(param))

    def _full_name(self):
        return "{}.{}".format(self._module_name, self.name)

    def __str__(self):
        cls_name = type(self).__name__
        return "<{} {}>".format(cls_name, self._full_name())


class TemplateClass(Template):
    def __init__(self, name, parent=None, **kwargs):
        Template.__init__(self, name, **kwargs)
        self.parent = parent

    def add_instantiation(self, param, cls):
        """ Adds instantiation. """
        # Do not double-register existing instantiation.
        if is_tpl_cls(cls):
            if self.parent != cls._tpl:
                # Do not permit any existing template.
                raise RuntimeError(
                    "Class already has template associated with it")
        # Nominal behavior.
        param = Template.add_instantiation(self, param, cls)
        # Update class information.
        cls._tpl = self
        cls.__name__ = self._get_instantiation_name(param)

    def add_instantiations(
            self, instantiation_func, param_list=None):
        if param_list is None:
            assert self.parent is not None
            param_list = self.parent.param_list
        Template.add_instantiations(self, instantiation_func, param_list)


def is_tpl_cls(cls):
    return hasattr(cls, '_tpl') and isinstance(cls._tpl, TemplateClass)


def is_tpl_of(cls, tpl):
    return is_tpl_cls(cls) and cls._tpl == tpl


class TemplateFunction(Template):
    def __init__(self, name, cls=None, **kwargs):
        Template.__init__(self, name, **kwargs)
        self._cls = cls

    def __str__(self):
        if self._cls is not None:
            return '<unbound TemplateFunction {}>'.format(self._full_name())
        else:
            return Template.__str__(self)

    def _full_name(self):
        return '{}.{}.{}'.format(self._module_name, self._cls.__name__, self.name)

    def __get__(self, obj, objtype):
        # Descriptor accessor.
        if obj is None:
            return self
        else:
            return _TemplateFunctionBound(self, obj)

    def __set__(self, obj, value):
        raise RuntimeError("Read-only property")


class _TemplateFunctionBound(object):
    def __init__(self, tpl, obj):
        assert tpl._cls is not None
        self._tpl = tpl
        self._obj = obj

    def __getitem__(self, param):
        unbound = self._tpl[param]
        bound = types.MethodType(unbound, self._obj, self._tpl._cls)
        return bound

    def __str__(self):
        return '<bound TemplateFunction {} of {}>'.format(
            self._tpl._full_name(), self._obj)
