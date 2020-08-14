"""
Provides an excessively minimal version of `nptyping`, but extends for use with
dataclasses, as well as pytorch.

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

import dataclasses as dc
import sys
from typing import Any, Callable, Type, TypeVar

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from .generic import (
    Dict,
    Dimension,
    Generic,
    List,
    N_,
    SizedList,
    Tuple,
    get_typing_name,
)


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
            if a == Any or b == Any:
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

        NDArray[(H, W), :] - example HxW image of any type
            NOTE: Because NDArray[(H, W)] is the same as NDArray[H, W], you
            must add a trailing comma. For simplicity (and to play well with
            the `black` formatter), you should add a simple slice.

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
        elif isinstance(x, slice) and x == slice(None):
            return GenericArray._IS_UNBOUND
        elif x is None:
            raise RuntimeError("None cannot be explicitly specified")
        else:
            return self._resolve_dtype_func(x)

    def _is_convertible_to_dtype(self, x):
        # Returns True if `x` can be converted to a dtype.
        # TODO(eric.cousineau): Make this more explicit.
        try:
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
            if self._is_convertible_to_dtype(x):
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

    def __hash__(self):
        return object.__hash__(self)

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


def _np_resolve_dtype(x):
    return np.dtype(x).type


# Defines an annotation for np.ndarray (which can be created using np.array).
NDArray = GenericArray(
    "NDArray", array_cls=np.ndarray, resolve_dtype=_np_resolve_dtype,
)


def _torch_resolve_dtype(x):
    # TODO(eric.cousineau): Determine better method.
    return torch.zeros(0, dtype=x).dtype


# Defines an annotation for torch.Tensor (which can be created using
# torch.tensor).
Tensor = GenericArray(
    "Tensor", array_cls=torch.Tensor, resolve_dtype=_torch_resolve_dtype,
)


class _Batch(Generic):
    """
    Batches any types, possibly transforming for torch.
    Follows suite with `default_collate`.

    Transformations:

        Numeric (int, float):
            Will be batched to Tensor[N_, DType]
            See below for DType transformations.
        Array:
            Will be converted to Tensor and batched (prepend N to shape).
            np.float32 and np.float64 will convert to torch.float32
        Tensor:
            Will simply prepend N to shape.
        Containers:
            List[T] -> List[Batch[T]]
            SizedList[M, T] -> SizedList[M, Batch[T]]
            Dict[K, V] -> Dict[K, Batch[V]]
            Tuple[T...] -> Tuple[Batch[T]...]
        Other Types:
            Shoveled into `SizedList[N_, T]`
        Struct:
            Recursive conversion.

    DType conversion:

        int -> torch.int32
        float -> torch.float32
        np.float64 -> torch.float32
    """

    # TODO(eric.cousineau): Provide way to dictacte dtype and image conversion?

    # TODO(eric.cousineau): Is there a better way to convert?
    _DTYPE_MAP = {
        int: torch.int32,
        float: torch.float32,
        np.float64: torch.float32,
        np.float32: torch.float32,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.uint8: torch.uint8,
        Any: Any,
    }

    def __init__(self):
        super().__init__(name="Batch", num_param=1)

    def _register_instantiation(self, T, cls):
        # This is necessary for pickle support:
        # https://bugs.python.org/issue35510
        # https://github.com/RobotLocomotion/drake/issues/11957#issuecomment-673743979  # noqa
        # If T is a type, register to its module. Otherwise, register to this
        # module.
        m = sys.modules[getattr(T, "__module__", __name__)]
        setattr(m, cls.__name__, cls)
        cls.__module__ = m.__name__

    def _make_instantiation(self, param):
        (T,) = param
        if T in self._DTYPE_MAP:
            dtype = self._DTYPE_MAP[T]
            return Tensor[N_, dtype]
        elif isinstance(T, GenericArray):
            shape = T.shape
            unbound = T.unbound()
            if N_ in shape:
                raise RuntimeError(f"Cannot batch - already batched! {T}")
            if unbound is NDArray:
                torch_dtype = self._DTYPE_MAP[T.dtype]
                batch_shape = (N_,) + shape
                return Tensor[batch_shape, torch_dtype]
            elif unbound is Tensor:
                new_shape = (N_,) + shape
                return Tensor[new_shape, T.dtype]
            else:
                assert False, T
        elif List.is_instantiation(T):
            (U,) = T.param
            return List[Batch[U]]
        elif SizedList.is_instantiation(T):
            M, U = T.param
            return SizedList[M, Batch[U]]
        elif Dict.is_instantiation(T):
            K, V = T.param
            return Dict[K, Batch[V]]
        elif Tuple.is_instantiation(T):
            Us = T.param
            NUs = tuple(Batch[U] for U in Us)
            return Tuple[NUs]
        elif dc.is_dataclass(T):
            fields = dc.fields(T)
            NT = dc.make_dataclass(
                cls_name=f"{self._name}[{T.__name__}]",
                fields=[(field.name, Batch[field.type]) for field in fields],
            )
            self._register_instantiation(T, NT)
            return NT
        elif isinstance(T, TypeVar):
            # TODO(eric.cousineau): Make this a named singleton?
            return object()
        else:
            return SizedList[N_, T]


# TODO(eric.cousineau): Rename to Collated?
Batch = _Batch()
T_ = TypeVar("T")


def _asdict_nonrecursive(x, fields):
    # We define this method because dc.asdict() is applied recursively (which
    # we do not want for `batch_collate`).
    assert dc.is_dataclass(x)
    out = dict()
    for field in fields:
        out[field.name] = getattr(x, field.name)
    return out


def batch_collate(batch: List[T_]) -> Batch[T_]:
    """Similar to `default_collate`, but handles dataclasses."""
    # TODO(eric.cousineau): Allow this to be changed?
    float_dtype = torch.float32
    numeric_cls = (np.ndarray, torch.Tensor, int, float)
    assert len(batch) > 0
    elem = batch[0]
    T = type(elem)
    for item in batch:
        U = type(item)
        assert U == T, f"{U} != {T}"
    if dc.is_dataclass(T):
        fields = dc.fields(T)
        batch_dict = [_asdict_nonrecursive(item, fields) for item in batch]
        collated_dict = batch_collate(batch_dict)
        collated = Batch[T](**collated_dict)
        return collated
    elif issubclass(T, dict):
        return {k: batch_collate([d[k] for d in batch]) for k in elem}
    elif issubclass(T, (list, tuple)):
        return T(batch_collate(samples) for samples in zip(*batch))
    elif issubclass(T, numeric_cls):
        collated = default_collate(batch)
        if isinstance(collated, torch.Tensor):
            if collated.dtype == torch.float64:
                collated = collated.type(float_dtype)
        return collated
    else:
        # Simply collect items into a list.
        # N.B. We must do this explicitly because `default_collate` will fail
        # fast on non-numeric and non-container types like `RigidTransform`.
        return list(batch)


def init_batch_collate_for_multiprocessing(dataset):
    """
    Ensures that we can use `batch_collate` in a multiprocessing context
    (e.g. torch...DataLoader).
    """
    if len(dataset) == 0:
        return
    batch_collate([dataset[0]])


DoubleArray = NDArray[np.float64]  # Same as NDArray[float]
LongArray = NDArray[np.int64]  # Same as NDArray[int]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]

DoubleTensor = Tensor[torch.float64]  # Same Tensor[float]
LongTensor = Tensor[torch.int64]  # Same Tensor[int]
FloatTensor = Tensor[torch.float32]
IntTensor = Tensor[torch.int32]

# Common dimensions.
# Generally, you should import this into other `typing_` modules.

# Width
W_ = Dimension("W")

# Height
H_ = Dimension("H")

# RGB channels (3)
C_ = Dimension("C", 3)

# Common aliases.
ImageArray = NDArray[(H_, W_, ...), :]
# TODO(eric.cousineau): Use `np.float32` for images.
RgbArray = NDArray[(H_, W_, C_), np.float64]
RgbIntArray = NDArray[(H_, W_, C_), np.uint8]
DepthArray = NDArray[(H_, W_), np.float32]
LabelArray = NDArray[(H_, W_), np.int32]
# NumPy- and Torch-friendly tensor for a mask.
MaskArray = NDArray[(H_, W_), np.uint8]
# Camera intrinsics for a pinhole model (K).
IntrinsicsArray = DoubleArray[(3, 3), :]

# N.B. We do not bake in batching here.
RgbTensor = Tensor[(C_, H_, W_), torch.float32]
# N.B. We purposely do not define `RgbIntTensor` as we do not have a use for it
# yet.
DepthTensor = Tensor[(H_, W_), torch.float32]
LabelTensor = Tensor[(H_, W_), torch.int32]
MaskTensor = Tensor[(H_, W_), torch.uint8]
IntrinsicsTensor = FloatTensor[(3, 3), :]


def print_typing_array_info(m):
    """
    Prints human-readable summary of basic typing info available in a
    module.
    """
    dimensions = dict()
    arrays = dict()
    for k, v in m.__dict__.items():
        if k.startswith("_"):
            continue
        if isinstance(v, Dimension):
            dimensions[k] = v
        elif isinstance(v, GenericArray):
            arrays[k] = v
    print(f"Typing info for {m.__name__}")
    print("  Dimensions:")

    def key(item):
        return item[0]

    for k, v in sorted(dimensions.items(), key=key):
        print(f"    {k}: {v}")
    print("  Arrays:")
    for k, v in sorted(arrays.items(), key=key):
        print(f"    {k}: {v}")
