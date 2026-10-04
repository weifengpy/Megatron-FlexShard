# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TransformerEngine's single grouped MoE weights (--moe-single-grouped-weight) for FlexShard.

TransformerEngine's GroupedLinear then stores its local experts' weights as one ``GroupedTensor``
parameter: a ``torch.Tensor`` wrapper over one packed buffer that bans slicing and other shape
ops, and that TransformerEngine's grouped GEMM requires. FlexShard shards a parameter by slicing
it and swaps plain tensors into the module, which would drop the wrapper. So:

- ``unwrap_single_grouped_params``, before FlexShard shards the model, replaces each
  ``GroupedTensor`` parameter with a plain parameter over the same buffer, with no copy: a weight
  as (experts * out, in), so FlexShard shards its rows, and a bias as (experts, out).
- Megatron's ``TEGroupedLinear`` wraps FlexShard's gathered plain tensor back into a
  ``GroupedTensor`` every forward (``grouped_view``), again with no copy, through an autograd
  function that returns the gradient as a plain tensor of the parameter's shape.
- Under gradient accumulation fusion, TransformerEngine writes the weight gradient into
  ``main_grad`` and returns none. FlexShard aliases ``main_grad`` at pre-backward, after the
  forward that created the view, so the view looks it up lazily through TransformerEngine's
  ``get_main_grad`` hook.
"""

import functools
from typing import Callable, Dict, Optional, Tuple

import torch

try:
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    HAVE_TE_GROUPED_TENSOR = True
except ImportError:
    HAVE_TE_GROUPED_TENSOR = False


class _AsGroupedTensor(torch.autograd.Function):
    """A ``GroupedTensor`` over a plain tensor, with no copy; its gradient goes back plain."""

    @staticmethod
    def forward(ctx, tensor, num_tensors, member_shape):
        # Under fusion TransformerEngine returns no weight gradient; don't let autograd turn
        # that into a full-size zero gradient for the plain parameter.
        ctx.set_materialize_grads(False)
        ctx.shape = tensor.shape
        return GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=num_tensors,
            tensor_shape=member_shape,
            rowwise_data=tensor.view(-1),
            dtype=tensor.dtype,
        )

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None
        if isinstance(grad, GroupedTensor):
            grad = grad.rowwise_data
        return grad.reshape(ctx.shape), None, None


def grouped_view(
    tensor: torch.Tensor, *, num_tensors: int, member_shape: Tuple[int, ...], main_grad: bool
) -> torch.Tensor:
    """The ``GroupedTensor`` TransformerEngine's grouped GEMM reads, over FlexShard's plain one."""
    view = _AsGroupedTensor.apply(tensor, num_tensors, member_shape)
    if main_grad:
        # TransformerEngine calls get_main_grad in backward (the hook Megatron-FSDP uses for its
        # lazily allocated main_grad).
        view.__fsdp_param__ = True
        view.get_main_grad = lambda: tensor.main_grad
    return view


@functools.lru_cache(maxsize=None)
def _grouped_tensor_attrs() -> frozenset:
    """``GroupedTensor``'s own instance attributes, which the plain parameter mustn't copy."""
    empty = GroupedTensor.make_grouped_tensor_from_rowwise_data(
        num_tensors=1, tensor_shape=(1,), rowwise_data=torch.empty(1, device="cuda")
    )
    return frozenset(vars(empty)) | {"_is_param"}


def unwrap_single_grouped_params(module: torch.nn.Module) -> int:
    """Replace each ``GroupedTensor`` weight or bias of a TransformerEngine grouped linear with a
    plain parameter over the same buffer, keeping Megatron's parameter attributes, and record the
    view each forward rebuilds. Returns how many it replaced."""
    if not HAVE_TE_GROUPED_TENSOR:
        return 0
    count = 0
    for submodule in module.modules():
        views: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for name in ("weight", "bias"):
            param: Optional[torch.Tensor] = submodule._parameters.get(name)
            if not isinstance(param, GroupedTensor):
                continue
            assert param.quantizer is None and param.columnwise_data is None, (
                f"FlexShard supports only high-precision single grouped {name}s, got "
                f"{type(submodule).__name__}.{name} with quantized storage."
            )
            member_shapes = {tuple(shape) for shape in param.tensor_shapes}
            assert len(member_shapes) == 1, f"Experts' {name}s differ in shape: {member_shapes}"
            (member_shape,) = member_shapes
            num_tensors = param.num_tensors
            if name == "weight":
                rows, cols = member_shape
                shape = (num_tensors * rows, cols)
            else:
                shape = (num_tensors, *member_shape)
            plain = torch.nn.Parameter(
                param.rowwise_data.view(shape), requires_grad=param.requires_grad
            )
            for key, value in vars(param).items():
                if key not in _grouped_tensor_attrs():
                    setattr(plain, key, value)
            submodule._parameters[name] = plain
            views[name] = functools.partial(
                grouped_view,
                num_tensors=num_tensors,
                member_shape=member_shape,
                main_grad=name == "weight",
            )
            count += 1
        if views:
            submodule.grouped_param_views = views
    return count
