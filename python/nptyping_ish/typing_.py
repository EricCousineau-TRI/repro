"""
Provides an excessively minimal version of `nptyping`.

Notable differences between this module:
- These types are not metaclasses.
- Rather than allowing things like `(Any, ...)` or `(N, ...)`, we only admit
`(...,)` for variable-dimension arrays.

nptyping looks great, dips into both metaclasses and some functional-ish
programming (by use of Union, etc., with pattern matching).
I (Eric) am not that smart to fully understand it, hence this basic
reimplementation :(

This should be thrown away once the following issues are resolved:
- https://github.com/ramonhagenaars/nptyping/issues/12
- https://github.com/ramonhagenaars/nptyping/issues/30

Should also monitor alternatives:
- https://github.com/ramonhagenaars/nptyping/issues/27
"""

from typing import Any, Callable, Tuple, Type, Union

import numpy as np

try:
    import torch

    _has_torch = True
except ModuleNotFoundError:
    _has_torch = False


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
        param_str = ", ".join(_get_name(x) for x in param)
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

    def __init__(self, name: str, *, num_param: Union[Tuple[int], int]):
        self._name = name
        if not isinstance(num_param, tuple):
            num_param = (num_param,)
        self._num_param = num_param
        self._instantiations = {}

    def _resolve_param(self, param: Tuple):
        """
        Resolves parameters. Override this in subclasses.
        """
        if len(param) in self._num_param:
            raise RuntimeError(
                f"{self} can only accept {self._num_param} parameter(s)"
            )
        return param

    def __getitem__(self, param: Union[Any, Tuple]):
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


def _compare_shape(shape_spec: Tuple, shape: Tuple):
    assert isinstance(shape_spec, tuple)
    assert isinstance(shape, tuple)
    if ... in shape_spec:
        assert shape_spec.count(...) == 1
        index = shape_spec.index(...)
        spec_before = shape_spec[:index]
        spec_after = shape_spec[index + 1 :]
        if len(spec_before) + len(spec_after) > len(shape):
            return False
        shape_before = shape[: len(spec_before)]
        if len(spec_after) > 0:
            shape_after = shape[-len(spec_after) :]
        else:
            shape_after = ()
        return _compare_shape(spec_before, shape_before) and _compare_shape(
            spec_after, shape_after
        )
    elif ... in shape:
        return _compare_shape(shape, shape_spec)
    else:
        if len(shape_spec) != len(shape):
            return False
        for a, b in zip(shape_spec, shape):
            if Any in (a, b):
                continue
            elif a != b:
                return False
        return True


class GenericArray(Generic):
    """
    Supports specifying shape and dtype.

    Similar contract to nptyping, but does *not* use metaclasses.

    Arrays can be specified as

        NDArray[Shape, DType]

    where Shape and DType are both optional.

    - Shape can be a tuple or a scalar, and any dimension whose value is Any
    means the dimension can be any (feasible) size. The Dimension class can be
    used as a simple named alias. The Ellipsis (...) means to allow any number
    of sizes at that point. At most, one ellipsis can be specified.

    - DType can be either a DType of the given framework (e.g. np.float32,
    torch.float32), a DType-compatible type (int, float), or Any.

    The default resolved shape and dtype is NDArray[..., Any], meaning any
    shape (including scalars), and any dtype.

    Some more examples:

        NDArray[Any] - Partially bound GenericArray with Any dtype (and
            effectively any shape).
        NDArray[..., Any] - Bound GenericArray with any dtype and shape.
        NDArray[10, np.int64] - Bound GenericArray with int64 and shape (10,).
        NDArray[(10,), np.int64] - Same as above.
        NDArray[10][np.int64] - Same as above.
        NDArray[np.int64][10] - Same as above.

        NDArray[(H, W), np.float32] - example float HxW depth image
        NDArray[(H, W, C), np.float32] - example float HxWxC RGB image

        NDArray[(H, W),] - example HxW image of any type
            NOTE: Because NDArray[(H, W)] is the same as NDArray[H, W], you
            must add the trailing comma!

    An GenericArray instance can be "unbound", "bound", or "partially bound":

    - "unbound" implies that you can still specify both shape and type.
    - "partially bound" implies that you cannot specify one of shape or dtype
    (one is already specified).
    - "bound" implies that you cannot specify anything.

    Please see the unittests for additional contracts to expect.

    Out of scope:
    - Record dtypes specified as tuples.
    """

    _DEFAULT_SHAPE = (...,)
    _DEFAULT_DTYPE = Any
    # Singleton.
    _IS_UNBOUND = object()

    def __init__(
        self,
        name: str,
        *,
        array_cls: Type,
        resolve_dtype: Callable,
        _shape=_IS_UNBOUND,
        _dtype=_IS_UNBOUND,
        _unbound=None,
    ):
        super().__init__(name, num_param=(1, 2))
        self._name = name
        self._array_cls = array_cls
        self._resolve_dtype_func = resolve_dtype
        self._shape = _shape
        self._dtype = _dtype

        # Rename to `__origin__`?
        self._unbound = _unbound
        if self._unbound is not None:
            # Connect the unbound (top-level) instantiation cache here.
            self._instantiations = self._unbound._instantiations

    def is_shape_bound(self) -> bool:
        """Returns whether Shape has been specified."""
        return self._shape is not GenericArray._IS_UNBOUND

    @property
    def shape(self) -> Tuple:
        """Returns either the bound shape, or the default shape if unbound."""
        if self.is_shape_bound():
            return self._shape
        else:
            return self._DEFAULT_SHAPE

    def is_dtype_bound(self) -> bool:
        """Returns whether or not the DType has been specified."""
        return self._dtype is not GenericArray._IS_UNBOUND

    @property
    def dtype(self):
        """Returns either the bound DType, or the default DType if unbound."""
        if self.is_dtype_bound():
            return self._dtype
        else:
            return self._DEFAULT_DTYPE

    def unbound(self) -> "GenericArray":
        """
        Returns unbound GenericArray, meaning you can specify dtype and shape.

        For more info, see:
        https://github.com/ramonhagenaars/nptyping/issues/31
        """
        if self._unbound is None:
            return self
        else:
            return self._unbound

    def bound(self) -> "GenericArray":
        """
        Returns fully bound GenericArray, meaning you can't specify dtype and
        shape.

        For more info, see:
        https://github.com/ramonhagenaars/nptyping/issues/31
        """
        unbound = self.unbound()
        return unbound[self.shape, self.dtype]

    def _resolve_shape(self, x):
        if x is GenericArray._IS_UNBOUND:
            return GenericArray._IS_UNBOUND
        elif x is None:
            raise RuntimeError("None cannot be explicitly specified")
        elif isinstance(x, tuple):
            assert x.count(...) in (
                0,
                1,
            ), "Can only have 0 or 1 instances of ..."
            return x
        else:
            return (x,)

    def _resolve_dtype(self, x):
        if x is GenericArray._IS_UNBOUND or x is Any:
            return x
        elif x is None:
            raise RuntimeError("None cannot be explicitly specified")
        else:
            return self._resolve_dtype_func(x)

    def _is_maybe_dtype(self, x):
        try:
            # TODO(eric.cousineau): Make this more explicit?
            self._resolve_dtype_func(x)
            return True
        except TypeError:
            return False

    def _resolve_shape_and_dtype(self, param):
        maybe_shape = GenericArray._IS_UNBOUND
        maybe_dtype = GenericArray._IS_UNBOUND
        if len(param) == 1:
            (x,) = param
            if x is None:
                raise RuntimeError("None cannot be explicitly specified")
            if self._is_maybe_dtype(x):
                maybe_dtype = x
            else:
                maybe_shape = x
        else:
            assert len(param) == 2, repr(param)
            maybe_shape, maybe_dtype = param
        shape = self._resolve_shape(maybe_shape)
        dtype = self._resolve_dtype(maybe_dtype)
        return (shape, dtype)

    def _resolve_param(self, param):
        # Overrides Generic._resolve_param, and resolves parameters specified
        # in Generic.__getitem__.
        if self.is_shape_bound() and self.is_dtype_bound():
            raise RuntimeError("Shape and type already bound")
        shape, dtype = self._resolve_shape_and_dtype(param)
        if self.is_shape_bound():
            if shape is not GenericArray._IS_UNBOUND:
                raise RuntimeError(
                    f"Shape is bound in {self}; cannot specify {shape}"
                )
            shape = self._shape
        if self.is_dtype_bound():
            if dtype is not GenericArray._IS_UNBOUND:
                raise RuntimeError(
                    f"DType is bound in {self}; cannot specify {dtype}"
                )
            dtype = self._dtype
        return (shape, dtype)

    def _make_instantiation(self, param):
        # Intercep
        unbound = self.unbound()
        shape, dtype = param
        # TODO(eric.cousineau): Is there a better structure for this?
        return GenericArray(
            self._name,
            array_cls=self._array_cls,
            resolve_dtype=self._resolve_dtype,
            _shape=shape,
            _dtype=dtype,
            _unbound=unbound,
        )

    def __eq__(self, other: "GenericArray") -> bool:
        """Compares GenericArray types."""
        if not isinstance(other, GenericArray):
            return False
        return (
            self._name == other._name
            and self._array_cls == other._array_cls
            and self._compare_shape(other.shape)
            and self._compare_dtype(other.dtype)
        )

    def _compare_shape(self, shape):
        return _compare_shape(self.shape, shape)

    def _compare_dtype(self, dtype):
        dtype_spec = self.dtype
        dtype = self._resolve_dtype(dtype)
        if Any in (dtype_spec, dtype):
            return True
        return dtype_spec == dtype

    def check_array(self, array):
        if isinstance(array, self._array_cls):
            if self._compare_shape(array.shape) and self._compare_dtype(
                array.dtype
            ):
                return True
        return False

    def to_type(self, array) -> "GenericArray":
        """Converts an array to a GenericArray type annotation."""
        assert isinstance(array, self._array_cls), repr(array)
        unbound = self.unbound()
        return unbound[array.shape, array.dtype]

    def __repr__(self):
        if len(self.shape) == 1:
            shape_str = get_typing_name(self.shape[0])
        else:
            shape_str = ", ".join(get_typing_name(x) for x in self.shape)
            shape_str = f"({shape_str})"
        dtype_str = get_typing_name(self.dtype)
        return f"{self._name}[{shape_str}, {dtype_str}]"


class Dimension:
    """
    Represents a simple named dimension.

    Comparison on this type (hashing, equality) is done directly on the value.
    """

    def __init__(self, name: str, value=Any):
        assert len(name) > 0
        self._name = name
        self._value = value

    def __eq__(self, other):
        if isinstance(other, Dimension):
            return self._value == other._value
        else:
            return self._value == other

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        return self._name


def _np_resolve_dtype(x):
    return np.dtype(x).type


# Defines an annotation for np.ndarray (which can be created using np.array).
NDArray = GenericArray(
    "NDArray", array_cls=np.ndarray, resolve_dtype=_np_resolve_dtype,
)
DoubleArray = NDArray[np.float64]  # Same as NDArray[float]
LongArray = NDArray[np.int64]  # Same as NDArray[int]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]

if _has_torch:

    def _torch_resolve_dtype(x):
        # Dunno a better way.
        return torch.zeros(0, dtype=x).dtype

    # Defines an annotation for torch.Tensor (which can be created using
    # torch.tensor).
    Tensor = GenericArray(
        "Tensor", array_cls=torch.Tensor, resolve_dtype=_torch_resolve_dtype,
    )
    DoubleTensor = Tensor[torch.float64]  # Same Tensor[float]
    LongTensor = Tensor[torch.int64]  # Same Tensor[int]
    FloatTensor = Tensor[torch.float32]
    IntTensor = Tensor[torch.int32]
