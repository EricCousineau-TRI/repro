#!/usr/bin/env python

# Template definitions.

def _tpl_name(name, param):
    return '{}[{}]'.format(name, ', '.join(param))


class Template(object):
    def __init__(self, name, param_default=None):
        self.name = name
        self._param_names = param_names
        self._param_default = param_default
        self._instantiations = {}

    def __getitem__(self, *param):
        """ Gets concrete class associate with the given arguments.

        If called with [[]], then returns the default instantiation. """
        if len(param) == 1 and param[0] == []:
            assert self._param_default is not None
            param =self._param_default
        cls = self._instantiations[param]
        return cls

    def add_instantiation(self, param, cls):
        """ Adds instantiation. """
        # Do not double-register existing instantiation.
        if is_tpl_cls(cls):
            self._check_tpl_cls(cls)
        # Ensure that we do not already have this tuple.
        assert param not in self._instantiations, "Param tuple already registered"
        # Add it.
        self._instantiations[param] = cls
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

    def add_instantiation_factory(self, cls_factory, param_list=None):
        if param_list is None:
            param_list = self._parent._instantiations.keys()
        for param in param_list:
            cls = cls_factory(*param)
            # Sanity check.
            base_cls = self._parent(*param)
            assert issubclass(cls, base_cls)
            self.add_instantiation(param, cls)

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
