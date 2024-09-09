import copy
import functools
import pickle
from pathlib import Path
from pprint import pprint


def _maybe_asdict(v):
    asdict = getattr(v, "asdict", None)
    if asdict:
        return asdict()
    else:
        return v


class GenericPickleObject:
    def __init__(self, *args, **kwargs):
        assert False, "This should only be created via pickling."

    def __setstate__(self, state):
        # Set by factory.
        assert self.module_name is not None
        assert self.class_name is not None
        self.state = state

    def asdict(self):
        state = copy.copy(self.state)
        if isinstance(state, dict):
            state = {k: _maybe_asdict(v) for k, v in state.items()}
        elif isinstance(state, (list, tuple)):
            cls = type(state)
            state = cls(_maybe_asdict(v) for v in state)
        return {
            "module_name": self.module_name,
            "class_name": self.class_name,
            "state": state,
        }

    @functools.lru_cache
    @staticmethod
    def make_factory(module_name, class_name):
        # Allows us to keep single class instance, but sideload additional
        # metadata during unpickling.

        class TempFactory:
            @staticmethod
            def __new__(cls):
                obj = GenericPickleObject.__new__(GenericPickleObject)
                # N.B. Nominal unpickling relies on `__setstate__`, so we will
                # rely on that codepath. For that reason, we will use
                # `__new()__`, and set class ino attributes manually.
                obj.module_name = module_name
                obj.class_name = class_name
                return obj

        return TempFactory


class _GenericUnpickler(pickle.Unpickler):
    def __init__(self, *args, is_generic_class, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_generic_class = is_generic_class

    def find_class(self, module_name, name):
        if self._is_generic_class(module_name, name):
            return GenericPickleObject.make_factory(module_name, name)
        else:
            return super().find_class(module_name, name)


def pickle_load_generic(f, is_generic_class, **kwargs):
    unpickler = _GenericUnpickler(
        f, is_generic_class=is_generic_class, **kwargs
    )
    return unpickler.load()


def is_generic_class_example(module_name, class_name):
    return module_name == "__main__"  # From "old_code" example.


def main():
    parent_dir = Path(__file__).parents[1]
    pkl_file = parent_dir / "old_data.pkl"

    # This does not work - will try to load from old codepath.
    with open(pkl_file, "rb") as f:
        raised = False
        try:
            obj = pickle.load(f)  # Does not work.
        except AttributeError as e:
            raised = True
            assert "Can't get attribute 'MyObject'" in str(e)
        finally:
            assert raised

    # This allows us to load undefind objects in a more generic fashion.
    with open(pkl_file, "rb") as f:
        obj = pickle_load_generic(f, is_generic_class=is_generic_class_example)

    pprint(obj.asdict())


assert __name__ == "__main__"
main()
