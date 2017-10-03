#!/usr/bin/env python

# Template definitions.

def _params_str(names, values):
    items = ''
    for (name, value) in zip(names, values):
        items.append('{}_{}'.format(name, value))
    return '__'.join(items)

def _tpl_name(name, param_names, params):
    return '{}__{}'.format(name, _params_str(param_names, params))

def _params_resolve(params, param_names):
    if isinstance(params, dict):
        return (params[name] for name in param_names)
    else:
        return params

class Template(object):
    def __init__(self, name, param_names, param_defaults=None):
        self.name = name
        assert isinstance(param_names, tuple)
        assert len(param_names) > 0
        self._param_names = param_names
        self._param_defaults = param_defaults
        self._instantiations = {}

    def __call__(self, *args, **kwargs):
        """ Gets concrete class associate with the given arguments. """
        # Can only have `args[0]` or `kwargs`, but not both.
        # TODO(eric.cousineau): Permit multiple "holes" for defaults?
        # Not necessary for now.
        if len(args) > 0:
            assert len(args) == 1
            params = args[0]
        elif len(kwargs) > 0:
            params = _params_resolve(kwargs, self._param_names)
        else:
            assert self._param_defaults is not None
            params = self._param_defaults
        cls = self._instantiations[params]
        return cls

    def add_instantiation(params_in, cls, override_name=True):
        """ Adds instantiation. """
        params = _params_resolve(params_in, self._param_names)
        # Do not double-register.
        assert not is_tpl(cls)
        # Ensure that we do not already have this tuple.
        assert params not in self.instantiations
        # Add it.
        self.instantiations[params] 
        cls._tpl = self
        if override_name:
            cls.__name__ = _tpl_name(self.name, self.param_names, params)

def is_tpl_cls(cls):
    return has_attr(cls, '_tpl') and isinstance(cls._tpl, Template)

def is_tpl_of(cls, tpl):
    return is_tpl(cls) and cls._tpl == tpl

"""
BaseTpl = Template(
    name = 'Base',
    param_names = ('T', 'U'),  # Ordering is important.
    param_defaults = (float, long),
)
BaseTpl.add_instantiation(
    params = (),  # Can be map or tuple.
    cls = ...,
)
# Default instantiation.
Base = BaseTpl()



# Inheritance - Simple method, default parameters
class Child(Base):
    def __init__(self):
        ...


# Inerhitance - Fancy method, any parameters
def _Child(T=?, U=?):
    Base = BaseTpl(T, U)
    class Child(Base):
        ...
        def do_to(self, Tc, Uc):
            ...
    return Child

# Inherits `default`.
ChildTpl = create_tpl_child(
    name = 'Child',
    parent = BaseTpl,
    def_func = _Child,
)
Child = ChildTpl()

"""
