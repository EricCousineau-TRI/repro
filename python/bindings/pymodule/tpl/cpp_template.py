#!/usr/bin/env python

import types

# Template definitions.
from pymodule.tpl.cpp_types import type_registry, _get_module_from_stack

PARAM_DEFAULT = ([],)
_PARAM_DEFAULT_DEFAULT = 'first_registered'


class Template(object):
    def __init__(self, name, param_default=_PARAM_DEFAULT_DEFAULT, module_name=None):
        self.name = name
        if isinstance(param_default, tuple) or isinstance(param_default, list):
            param_default = self.param_canonical(param_default)
        self._param_default = param_default
        self.param_list = []
        # @note Consider using `StrictMap` if literals must exactly match.
        # (e.g. `0` and `False` should resolve to different instantiations).
        self._instantiation_map = {}
        if module_name is None:
            module_name = _get_module_from_stack()
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
        """Gets concrete class associate with the given arguments. """
        if not isinstance(param, tuple) or isinstance(param, list):
            # Handle scalar case.
            param = (param,)
        return self.get_instantiation(param)

    def get_instantiation(self, param=PARAM_DEFAULT, throw_error=True):
        """Gets the instantiation for the given parameters. """
        param = self._param_resolve(param)
        if throw_error:
            return self._instantiation_map[param]
        else:
            return self._instantiation_map.get(param)

    def _get_instantiation_name(self, param):
        param_str = map(type_registry.GetName, param)
        return '{}[{}]'.format(self.name, ', '.join(param_str))

    def add_instantiation(self, param, instantiation):
        """ Adds instantiation. """
        # Ensure that we do not already have this tuple.
        param = self.param_canonical(param)
        print(param)
        print(instantiation)
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
        # Add metadata to instantiation.
        cls._tpl = self
        cls._tpl_param = param
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
    pass


class TemplateMethod(TemplateFunction):
    def __init__(self, name, cls, **kwargs):
        TemplateFunction.__init__(self, name, **kwargs)
        self._cls = cls

    def __str__(self):
        return '<unbound TemplateMethod {}>'.format(self._full_name())

    def _full_name(self):
        return '{}.{}.{}'.format(self._module_name, self._cls.__name__, self.name)

    class _Bound(object):
        def __init__(self, tpl, obj):
            self._tpl = tpl
            self._obj = obj

        def __getitem__(self, param):
            unbound = self._tpl[param]
            bound = types.MethodType(unbound, self._obj, self._tpl._cls)
            return bound

        def __str__(self):
            return '<bound TemplateMethod {} of {}>'.format(
                self._tpl._full_name(), self._obj)

    def __get__(self, obj, objtype):
        # Descriptor accessor.
        if obj is None:
            return self
        else:
            return TemplateMethod._Bound(self, obj)

    def __set__(self, obj, value):
        raise RuntimeError("Read-only property")
