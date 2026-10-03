# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.distributed import ProcessGroup

try:
    from flex_shard import BucketSpec, MixedPrecisionPolicy, flex_shard
    from flex_shard.custom_placements.shard import per_param_placements
    from torch.distributed.device_mesh import DeviceMesh

    HAVE_FLEX_SHARD = True
except ImportError:
    HAVE_FLEX_SHARD = False

from ... import parallel_state
from ...models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from ...process_groups_config import ProcessGroupCollection
from ...tensor_parallel import ColumnParallelLinear
from ...transformer.transformer_config import TransformerConfig
from ...transformer.transformer_layer import TransformerLayer
from ...utils import log_single_rank
from ..data_parallel_base import _BaseDataParallel
from .flex_shard_data_parallel_config import FlexShardDataParallelConfig

logger = logging.getLogger(__name__)


class FlexShardDataParallel(_BaseDataParallel):
    """
    Shards parameters over the data-parallel group with FlexShard
    (https://github.com/meta-pytorch/flex_shard).

    Each bucket is one all-gather before use and one reduce-scatter after backward over the
    data-parallel group, with every parameter sharded along dim 0. Buckets follow forward
    order: the embedding, one bucket per TransformerLayer, any remaining parameter-owning
    modules (e.g. the final layernorm), and the output layer.

    With tensor parallelism, each rank shards its TP slices over its data-parallel group,
    which excludes its TP peers, so the model's own TP communication is unchanged.
    Layer-norm gradients that are partial over TP (sequence parallelism, qk_layernorm) are
    reduce-scattered over DP during backward, then all-reduced over TP after backward in one
    coalesced call (finalize_model_grads) on the local shards, 1/DP of each norm. Megatron DDP
    does the same on full main_grads. torchtitan's FSDP2 + TP instead all-reduces the full
    gradients over TP per block inside backward, before the DP reduce-scatter.

    After wrapping, ``module.parameters()`` yields local shards as plain tensors, so a
    non-distributed Megatron optimizer updates only this rank's shard. Gradients are reduced
    during backward and waited on at the end of backward, so ``finish_grad_sync`` has nothing
    left to do. Inside ``no_sync()`` (Megatron's ``no_sync_func``, wired with
    ``ddp_config.no_sync``), backwards skip the reduce-scatter and accumulate full gradients,
    in fp32 with ``grad_reduce_in_fp32``; the last microbatch's backward reduce-scatters them.
    With ``grad_reduce_in_fp32``, local-shard gradients are fp32 too, as the distributed
    optimizer keeps them.

    With gradient accumulation fusion, TransformerEngine's and Megatron's linear layers add
    weight gradients straight into ``param.main_grad`` and give autograd none. For buckets
    with such layers, FlexShard's pre-backward hook allocates each gathered parameter's
    gradient and aliases it as ``main_grad``, and its post-reduce hook drops the alias once
    the reduce-scatter has taken the gradient. The fused GEMMs thus accumulate into the
    gradient FlexShard reduce-scatters, across microbatches with no_sync.

    Args:
        config: Transformer config object.
        ddp_config: FlexShardDataParallelConfig object.
        module: Underlying model.
        disable_bucketing: Unused; accepted for interface compatibility with DDP.
        pg_collection: Optional ProcessGroupCollection; parameters are sharded over its
            ``dp_cp`` group.
        process_group: Optional ProcessGroup to shard over. Takes precedence over
            ``pg_collection``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: FlexShardDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        process_group: Optional[ProcessGroup] = None,
    ):
        assert HAVE_FLEX_SHARD, 'FlexShardDataParallel requires the flex_shard package.'

        super().__init__(config=config, module=module)
        self.ddp_config = ddp_config

        if process_group is not None:
            self.process_group = process_group
        elif pg_collection is not None:
            self.process_group = pg_collection.dp_cp
        else:
            # Migration fallback for callers that do not pass process groups.
            self.process_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        self.device_mesh = DeviceMesh.from_group(
            self.process_group, "cuda", mesh_dim_names=("dp",)
        )

        # FlexShard replaces each parameter with a local-shard tensor, dropping the Megatron
        # attributes (tensor_model_parallel, allreduce, ...) the optimizer relies on.
        custom_attrs = {
            name: dict(vars(param)) for name, param in self.module.named_parameters()
        }

        mp_policy = MixedPrecisionPolicy(
            reduce_dtype=torch.float32 if ddp_config.grad_reduce_in_fp32 else None
        )
        bucket_fqns = self._build_bucket_fqns()
        fused_fqns = self._fused_wgrad_fqns() if config.gradient_accumulation_fusion else set()
        uses_main_grad = [any(fqn in fused_fqns for fqn in fqns) for fqns in bucket_fqns]
        self.buckets = [
            BucketSpec(
                fqns,
                placement_fn=per_param_placements,
                mesh=self.device_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=(
                    ddp_config.reshard_after_forward and i < len(bucket_fqns) - 1
                ),
                pre_backward_hook=self._alias_main_grads if uses_main_grad[i] else None,
                post_reduce_hook=self._drop_main_grads if uses_main_grad[i] else None,
            )
            for i, fqns in enumerate(bucket_fqns)
        ]
        if ddp_config.grad_reduce_in_fp32:
            # FlexShard stores each local shard's gradient in the parameter's grad_dtype
            # (flex_shard#20), so bf16 parameters get fp32 shard gradients, which the
            # mixed-precision optimizer then uses as main gradients without a copy.
            for param in self.module.parameters():
                if param.is_floating_point() and param.dtype != torch.float32:
                    param.grad_dtype = torch.float32
        flex_shard(self.module, buckets=self.buckets)
        self.module.set_reshard_after_backward(ddp_config.reshard_after_backward)

        for name, param in self.module.named_parameters():
            for attr_name, attr_value in custom_attrs.get(name, {}).items():
                setattr(param, attr_name, attr_value)
            param.is_flex_shard_param = True

        log_single_rank(
            logger,
            logging.INFO,
            f"FlexShard: {len(self.buckets)} buckets over {self.device_mesh.size()} ranks, "
            f"reshard_after_forward={ddp_config.reshard_after_forward}, "
            f"no_sync={ddp_config.no_sync}, "
            f"reshard_after_backward={ddp_config.reshard_after_backward}, "
            f"main_grad buckets={sum(uses_main_grad)}, "
            f"local params={sum(p.numel() for p in self.module.parameters())}",
        )

    def _fused_wgrad_fqns(self) -> Set[str]:
        """FQNs of parameters whose modules add weight gradients into ``main_grad``."""
        return {
            f"{name}.{param_name}" if name else param_name
            for name, submodule in self.module.named_modules()
            if getattr(submodule, "fuse_wgrad_accumulation", False)
            or getattr(submodule, "gradient_accumulation_fusion", False)
            for param_name, _ in submodule.named_parameters(recurse=False)
        }

    def _alias_main_grads(self, named_params: List[Tuple[str, torch.nn.Parameter]]) -> None:
        """FlexShard pre-backward hook: expose each gathered parameter's gradient as main_grad.

        Fused gradient accumulation adds weight gradients into ``main_grad`` in place, so the
        gradient must exist before the bucket's backward. A missing one is allocated zeroed in
        the accumulation dtype, which autograd's gradients for unfused parameters also add into.
        Without gradient sync, FlexShard keeps it for the next microbatch.
        """
        for _, param in named_params:
            if not param.requires_grad:
                continue
            if param.grad is None:
                dtype = torch.float32 if self.ddp_config.grad_reduce_in_fp32 else param.dtype
                param.grad = torch.zeros(param.shape, dtype=dtype, device=param.device)
            param.main_grad = param.grad

    @staticmethod
    def _drop_main_grads(named_params: List[Tuple[str, torch.nn.Parameter]]) -> None:
        """FlexShard post-reduce hook: drop main_grad once the gradient it aliases is gone."""
        for _, param in named_params:
            vars(param).pop("main_grad", None)

    def _build_bucket_fqns(self) -> List[List[str]]:
        """Group parameter FQNs into buckets in forward (module registration) order."""
        params = dict(self.module.named_parameters())
        claimed = set()
        buckets: List[List[str]] = []

        def take(fqns):
            fqns = [fqn for fqn in fqns if fqn in params and fqn not in claimed]
            claimed.update(fqns)
            return fqns

        for name, submodule in self.module.named_modules():
            prefix = f"{name}." if name else ""
            if isinstance(submodule, (TransformerLayer, LanguageModelEmbedding, ColumnParallelLinear)):
                fqns = take(prefix + n for n, _ in submodule.named_parameters())
            else:
                fqns = take(prefix + n for n, _ in submodule.named_parameters(recurse=False))
            if fqns:
                buckets.append(fqns)
        assert claimed == set(params), f"Unbucketed parameters: {set(params) - claimed}"

        # FlexShard requires one parameter dtype per bucket.
        split_buckets = []
        for fqns in buckets:
            by_dtype: Dict[torch.dtype, List[str]] = {}
            for fqn in fqns:
                by_dtype.setdefault(params[fqn].dtype, []).append(fqn)
            split_buckets.extend(by_dtype.values())
        return split_buckets

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization: backwards inside it
        accumulate full gradients without reduce-scatter.
        """
        self.module.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            self.module.set_requires_gradient_sync(True)

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all local gradient shards by `scaling_factor`."""
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.mul_(scaling_factor)

    def finish_grad_sync(self, force_all_reduce=False):
        """
        No-op: FlexShard waits for all reduce-scatters at the end of backward.
        """
        pass
