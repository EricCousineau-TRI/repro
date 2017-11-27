value = 5

class ModuleShim(object):
    """ Provide a shim for automatically resolving extra variables.
    This can be used to deprecate import alias in modules, to simplify
    dependencies.

    @see https://stackoverflow.com/a/7668273/7829525 """

    def __init__(self, orig_module, extras):
        # 'extras` should not share variables with `orig_module` at
        # construction.
        common = set(extras.keys()).intersection(set(dir(orig_module)))
        assert len(common) == 0, (
            "Should not share variables at construction: {}".format(common))
        assert hasattr(orig_module, "__all__"), (
            "Please define `__all__` for this module.")
        # https://stackoverflow.com/a/16237698/7829525
        object.__setattr__(self, '_orig_module', orig_module)
        object.__setattr__(self, '_extras', extras)

    def __getattr__(self, name):
        m = self.__dict__["_orig_module"]
        if hasattr(m, name):
            return getattr(m, name)
        elif name in self._extras:
            value = self._extras[name]()
            setattr(self._orig_module, name, value)
            return value
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        setattr(self._orig_module, name, value)

    @classmethod
    def install(cls, extras, name = __name__):
        import sys
        old_module = sys.modules[name]
        new_module = cls(old_module, extras)
        sys.modules[name] = new_module

def _extra():
    print("`import mod; mod.extra` will soon be deprecated." +
          "Please use `import mod.extra` instead.")
    return 10

__all__ = locals().keys() + ["extra"]

ModuleShim.install({
    "extra": _extra
})
