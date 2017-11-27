import sys
import traceback

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
            try:
                value = self._handler(name)
            except AttributeError as e:
                if e.message:
                    raise e
                else:
                    raise AttributeError(
                        "'module' object has no attribute '{}'".format(name))
            setattr(m, name, value)
            return value

    def __setattr__(self, name, value):
        # Redirect writes to the original module.
        setattr(self._orig_module, name, value)

    @classmethod
    def install(cls, name, handler):
        """ Hook into module's attribute accessors and mutators. """
        old_module = sys.modules[name]
        new_module = cls(old_module, handler)
        sys.modules[name] = new_module


def in_from_import():
    stack = traceback.extract_stack()
    print("\n".join(map(str, stack)))
    FUNC = 2
    assert stack[-2][2] == "_handler"
    assert stack[-3][2] == "__getattr__"
    if stack[-4][3].lstrip().startswith("from "):
        return True
    else:
        return False

def _handler(name):
    if name == "extra":
        print(in_from_import())
        sys.stderr.write(
            "`import mod; mod.extra` will soon be deprecated. " +
            "Please use `import mod.extra` instead.\n")
        return 10
    else:
        raise AttributeError()


__all__ = locals().keys() # + ["extra"]
ModuleShim.install(__name__, _handler)
