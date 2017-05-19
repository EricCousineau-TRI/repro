#!/usr/bin/env python

# @ref http://stackoverflow.com/a/28516918
import os
import types
import importlib

def reload_package(package):
    assert(hasattr(package, "__package__"))
    fn = package.__file__
    fn_dir = os.path.dirname(fn) + os.sep
    module_visit = {fn}
    del fn

    def reload_recursive_ex(module):
        reload(module)
        for module_child in vars(module).values():
            if isinstance(module_child, types.ModuleType):
                fn_child = getattr(module_child, "__file__", None)
                if fn_child is not None \
                        and fn_child.startswith(fn_dir) \
                        and fn_child not in module_visit:
                    module_visit.add(fn_child)
                    reload_recursive_ex(module_child)
    return reload_recursive_ex(package)

if __name__ == "__main__":
    # example use
    print "Corrupt"
    os.path = None
    print "Reload"
    reload(os)
    import os
    assert os.path is not None
    print "Good"
