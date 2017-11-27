import sys

value = 5


class ModuleShim(object):
    """ Provide a shim for automatically resolving extra variables.
    This can be used to deprecate import alias in modules, to simplify
    dependencies.

    @see https://stackoverflow.com/a/7668273/7829525 """

    def __init__(self, orig_module, handler):
        assert hasattr(orig_module, "__all__"), (
            "Please define `__all__` for this module.")
        # https://stackoverflow.com/a/16237698/7829525
        object.__setattr__(self, '_orig_module', orig_module)
        object.__setattr__(self, '_handler', handler)

    def __getattr__(self, name):
        # Use the original module if possible.
        m = self._orig_module
        if hasattr(m, name):
            return getattr(m, name)
        else:
            # Otherwise, use the handler, and store the result.
            value = self._handler(name)
            setattr(m, name, value)
            return value

    def __setattr__(self, name, value):
        # Redirect writes to the original module.
        setattr(self._orig_module, name, value)

    @classmethod
    def install(cls, name, handler):
        """ Hook into module's attribute accessors and mutators. """
        import sys
        old_module = sys.modules[name]
        new_module = cls(old_module, handler)
        sys.modules[name] = new_module


def _handler(name):
    if name == "extra":
        sys.stderr.write(
            "`import mod; mod.extra` will soon be deprecated. " +
            "Please use `import mod.extra` instead.\n")
        return 10
    else:
        raise AttributeError(
            "'module' object has no attribute '{}'".format(name))


__all__ = locals().keys() + ["extra"]
ModuleShim.install(__name__, _handler)
