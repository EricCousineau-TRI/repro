import dataclasses as dc
from typing import Any, Union


def get_typing_name(x: Any):
    """
    Gets the name for a class or class-like object.

    For brevity, this will strip off module names for known class-like objects
    (e.g. torch dtypes, or typing objects).
    """
    if isinstance(x, type):
        return x.__name__
    elif x is Ellipsis:
        return "..."
    else:
        s = repr(x)
        typing_prefix = "typing."
        torch_prefix = "torch."
        if s.startswith(typing_prefix):
            return s[len(typing_prefix) :]
        elif s.startswith(torch_prefix):
            return s[len(torch_prefix) :]
        else:
            return s


class _GenericInstantiation:
    """Indicates a simple instantiation of a Generic."""

    def __init__(self, generic, param):
        # TODO(eric.cousineau): Rename to `__origin__`?
        self._generic = generic
        self._param = param
        param_str = ", ".join(get_typing_name(x) for x in param)
        self._full_name = f"{generic._name}[{param_str}]"

    @property
    def param(self):
        """Parameters for given instantiation."""
        # TODO(eric): Rename to `__args__`?
        return self._param

    def __repr__(self):
        return self._full_name


class Generic:
    """
    Provides a way to denote generic classes and cache "instantiations".

    The ``typing`` module in Python provides generics like this; however, the
    API does not admit easy inspection (whyeee????!!!), at least in Python 3.6
    and 3.8, thus we reinvent a smaller wheel.
    """

    def __init__(self, name: str, *, num_param: Union["Tuple[int]", int]):
        self._name = name
        if not isinstance(num_param, tuple):
            num_param = (num_param,)
        self._num_param = num_param
        self._instantiations = {}

    def _resolve_param(self, param: "Tuple"):
        """
        Resolves parameters. Override this in subclasses.
        """
        if self._num_param != (Any,) and len(param) not in self._num_param:
            raise RuntimeError(
                f"{self} can only accept {self._num_param} parameter(s). "
                f"{len(param)} given."
            )
        return param

    def __getitem__(self, param: Union[Any, "Tuple"]):
        """Creates or retrieves a cached instantiation."""
        if not isinstance(param, tuple):
            param = (param,)
        param = self._resolve_param(param)
        instantiation = self._instantiations.get(param)
        if instantiation is None:
            instantiation = self._make_instantiation(param)
            self._instantiations[param] = instantiation
        return instantiation

    def _make_instantiation(self, param):
        return _GenericInstantiation(self, param)

    def is_instantiation(self, instantiation):
        return instantiation in self._instantiations.values()

    def __repr__(self):
        return f"<{type(self).__name__} {self._name}>"


# Type for FieldHint to indicate structure for list/dict.
# N.B. See my (Eric's) complaints in `Generic` for why this isn't using
# `typing.List`, `typing.Dict`.
List = Generic("List", num_param=1)
Dict = Generic("Dict", num_param=2)
# TODO(eric.cousineau): Try to use `typish` and avoid needing to redefine this.
Tuple = Generic("Tuple", num_param=Any)

# This should be specified as SizedList[N, T].
# TODO(eric.cousineau): Make this more functional.
SizedList = Generic("SizedList", num_param=2)


def pformat_dataclass(cls):
    assert dc.is_dataclass(cls)
    fields = dc.fields(cls)
    assert len(fields) > 0
    out = f"@dataclass\n"
    out += f"class {get_typing_name(cls)}:\n"
    for field in fields:
        out += f"    {field.name}: {get_typing_name(field.type)}"
        if (
            field.default is not dc.MISSING
            or field.default_factory is not dc.MISSING
        ):
            out += " = <default>"
        out += "\n"
    return out


class Dimension:
    """
    Represents a simple named dimension (e.g. for an array's shape, a list's
    size, etc).

    Comparison on this type (hashing, equality) is done directly on the value
    when only the value is being compared. However, for hashing and comparison
    against other Dimension's, we use `object.__hash__` (based on `id()`). This
    prevents inadvertent mixing of different "symbolic" dimensions among
    modules.

        Dimension("D") == Any
        Dimension("D") != Dimension("D")
        Dimension("D2", 2) == 2
        Dimension("D2", 2) != Dimension("D2", 2)
    """

    # TODO(eric.cousineau): Use symbolic stuff from Drake if we need math?

    def __init__(self, name: str, value=Any):
        assert len(name) > 0
        self._name = name
        self._value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        if isinstance(other, Dimension):
            return self is other
        else:
            return self._value == other

    def __hash__(self):
        return object.__hash__(self)

    def __repr__(self):
        return self._name


# General batch size (used in ``torch.nn`` documentation).
# `B` may be better, but consistency :(
N_ = Dimension("N")


class _SoA(Generic):
    """Transforms a class to a struct of arrays."""

    def __init__(self):
        super().__init__(name="SoA", num_param=1)

    def _make_instantiation(self, param):
        assert len(param) == 1
        (T,) = param
        fields = dc.fields(T)
        assert dc.is_dataclass(T), T
        cls = dc.make_dataclass(
            cls_name=f"{self._name}[{T.__name__}]",
            fields=[
                (field.name, SizedList[N_, field.type]) for field in fields
            ],
        )
        return cls


class _AoS(Generic):
    def __init__(self):
        super().__init__(name="AoS", num_param=1)

    def _make_instantiation(self, param):
        (T,) = param
        return SizedList[N_, T]


AoS = _AoS()
SoA = _SoA()
