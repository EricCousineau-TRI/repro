import functools

import numpy as np
import torch

from pydrake.autodiffutils import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
)


class DrakeTorchFunction(torch.autograd.Function):
    """
    Given a Drake function `func`, this provides a PyTorch operation that can
    be used in a computation graph for backprop (e.g., optimization).

    See `drake_torch_function` for a more user-friendly approach.

    Let:
        x be the inputs for this function
        f(x) represent `forward(x)`
        ∂L/∂f be the loss gradient "after" this function in the graph (but
            "before" in terms of backprop traversal)
        ∂L/∂x be the loss gradient being backprop'ed "before" this function in
            the graph (but "after" in terms of backprop traversal)

    Locally (via Drake and AutoDiffXd), we can compute ∂f/∂x.
    When .backward() is called, it is given ∂L/∂f, and we compute ∂L/∂x via the
    chain rule:

        ∂L/∂x = ∂L/∂f ⋅ ∂f/∂x

    This class provides to handle combine multiple arguments (of arbitrary
    shape) and arbitrary need of gradients.

    Note:
        This is *not* batched. It is the responsibility of the calling class to
        handle batching / unbatching.

    For more details about torch.autograd.Function, see:
    https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html#advanced-topic-more-autograd-detail-and-the-high-level-api
    """

    @staticmethod
    def forward(ctx, func, *args):
        (
            args_new,
            indices_ad,
            shapes_ad,
            device,
            dtype,
        ) = torch_function_args_to_autodiff(args)
        f_ad = func(*args_new)
        f, df_dx = autodiff_to_torch_value_and_grad(f_ad, device, dtype)
        # Save local gradients.
        ctx.save_for_backward(df_dx)
        # Save metadata to unpack gradients in the same shape as our inputs.
        nargs = len(args)
        ctx.extra = (nargs, indices_ad, shapes_ad)
        return f

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dL_df):
        # Chain rule against the entire gradient.
        dL_df = dL_df.reshape(-1)
        (df_dx,) = ctx.saved_tensors
        dL_dx = dL_df @ df_dx
        # Reshape gradients to match ordering of original args,
        # dL_dw -> grad_w
        (nargs, indices_ad, shapes_ad) = ctx.extra
        grads_args = torch_function_reshape_grads(
            dL_dx, nargs, indices_ad, shapes_ad
        )
        # Add (no) grad return for `func` input argument.
        return (None,) + grads_args


def drake_torch_function(func):
    """
    Decorates a function `func` with a .torch attribute to allow forward eval
    and backprop in a PyTorch computation graph.

    For example:

        @drake_torch_function
        def my_drake_function(plant, context, q, v):
            plant.SetPositions(context, q)
            plant.SetVelocities(context, v)
            return plant.SomeOperation(...)

        ...

        class MyDrakeModule(nn.Module):
            def __init__(self, plant):
                assert isinstance(plant, MultibodyPlant_[AutoDiffXd])
                self.plant = plant
                self.context = self.plant.CreateDefaultContext()

            def forward(self, q, v):
                return my_drake_function.torch(self.plant, self.context, q, v)
    """
    func.torch = functools.partial(DrakeTorchFunction.apply, func)
    return func


def torch_function_args_to_autodiff(args):
    """
    Converts `args` to `args_new`, where any argument that is a torch
    tensor requiring gradients are converted to AutoDiffXd elements.

    Returns: (args_new, indices_ad, shapes_ad, device, dtype)
        args_new: Remapped arguments that can be used with Drake.
        indices_ad: Indices of AutoDiffXd arguments.
        shapes_ad: Shapes of AutoDiffXd arguments.

    Note:
        While we could also convert torch tensors that don't need gradients to
        AutoDiffXd arrays, it seems like C++ / pybind11 arguments are OK
        accepting torch tensors.
    """
    # shapes: Only for arguments requiring gradients.
    device = None
    dtype = None
    args_new = list(args)
    indices_ad = []
    args_ad = []
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if arg.requires_grad:
                indices_ad.append(i)
                args_ad.append(arg)
            else:
                arg_new = arg.detach().cpu().numpy()
                args_new[i] = arg_new
            if device is None:
                device = arg.device
                dtype = arg.dtype
    args_ad, shapes_ad = torch_to_initialize_autodiff_list(args_ad)
    for i, arg_ad in zip(indices_ad, args_ad):
        args_new[i] = arg_ad
    return args_new, indices_ad, shapes_ad, device, dtype


def torch_function_reshape_grads(dL_dx, nargs, indices_ad, shapes_ad):
    """
    Reshapes the gradients ∂L/∂x to {∂L/∂xᵢ}ᵢ, where xᵢ retains the same shape
    as the input argument passed to `torch_function_args_to_autodiff`.
    """
    grads_all = [None] * nargs
    grads_ad = unflatten_arrays(dL_dx, shapes_ad)
    for i, grad_ad in zip(indices_ad, grads_ad):
        grads_all[i] = grad_ad
    return tuple(grads_all)


def autodiff_to_value(v):
    """Similar to ExtractValue, but retains original shape."""
    shape = v.shape
    return ExtractValue(v).reshape(shape)


def autodiff_to_value_and_grad(v):
    """
    Extracts both value and gradient from AutoDiffXd array `v`.
    """
    value = autodiff_to_value(v)
    grad = ExtractGradient(v)
    return value, grad


def autodiff_to_torch_value_and_grad(v, device, dtype):
    """
    Converts AutoDiffXd array `v` to torch tensors of both value and gradient.
    """
    value, grad = autodiff_to_value_and_grad(v)
    value = torch.from_numpy(value).to(device, dtype)
    grad = torch.from_numpy(grad).to(device, dtype)
    return value, grad


def flatten_tensors(vs):
    """
    Given a list of tensors `vs`, flattens them into a 1D tensor and provides
    the shapes of the original shapes.
    """
    v_flat_list = []
    shapes = []
    for v in vs:
        v_flat = v.reshape(
            -1,
        )
        shape = v.shape
        v_flat_list.append(v_flat)
        shapes.append(shape)
    if len(v_flat_list) > 0:
        vs_flat = torch.cat(v_flat_list)
    else:
        vs_flat = torch.tensor([])
    return vs_flat, shapes


def unflatten_arrays(vs_flat, shapes):
    """
    Reverse of flatten_tensors(), but for NumPy arrays.
    """
    lens = [int(np.prod(shape)) for shape in shapes]
    vs = []
    i = 0
    for shape in shapes:
        len_i = int(np.prod(shape))
        v_i = vs_flat[i : i + len_i].reshape(shape)
        vs.append(v_i)
        i += len_i
    assert i == vs_flat.shape[0]
    return vs


def torch_to_initialize_autodiff_list(vs):
    for v in vs:
        assert v.requires_grad
    vs_flat, shapes = flatten_tensors(vs)
    # TODO(eric.cousineau): This is a bit inefficient, in that it creates a
    # large NxN matrix, then chunks it up.
    vs_flat = InitializeAutoDiff(vs_flat.detach().cpu().numpy())
    vs = unflatten_arrays(vs_flat, shapes)
    return vs, shapes
