# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Dict, List, Optional

import torch
from torch.distributed import ProcessGroup

try:
    from flex_shard import BucketSpec, MixedPrecisionPolicy, flex_shard
    from flex_shard.custom_placements.shard import per_param_placements
    from torch.distributed.device_mesh import DeviceMesh

    HAVE_FLEX_SHARD = True
except ImportError:
    HAVE_FLEX_SHARD = False

from .. import parallel_state
from ..models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from ..process_groups_config import ProcessGroupCollection
from ..tensor_parallel import ColumnParallelLinear
from ..transformer.transformer_config import TransformerConfig
from ..transformer.transformer_layer import TransformerLayer
from ..utils import log_single_rank
from .data_parallel_base import _BaseDataParallel
from .flex_shard_data_parallel_config import FlexShardDataParallelConfig

logger = logging.getLogger(__name__)


class FlexShardDataParallel(_BaseDataParallel):
    """
    Shards parameters over the data-parallel group with FlexShard
    (https://github.com/meta-pytorch/flex_shard).

    Each bucket is one all-gather before use and one reduce-scatter after backward over the
    data-parallel group, with every parameter sharded along dim 0. Buckets follow forward
    order: the embedding, one bucket per TransformerLayer, any remaining parameter-owning
    modules (e.g. the final layernorm), and the output layer. FlexShard hooks each bucket on
    the deepest module owning all of its parameters, so a bucket spanning several layers
    would hook their ModuleList, whose forward never runs.

    After wrapping, ``module.parameters()`` yields local shards as plain tensors, so a
    non-distributed Megatron optimizer updates only this rank's shard. Gradients are reduced
    during backward and waited on at the end of backward, so ``finish_grad_sync`` has nothing
    left to do. FlexShard has no no-sync mode: every microbatch's backward reduce-scatters.

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
        self.buckets = [
            BucketSpec(
                fqns,
                placement_fn=per_param_placements,
                mesh=self.device_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=(
                    ddp_config.reshard_after_forward and i < len(bucket_fqns) - 1
                ),
            )
            for i, fqns in enumerate(bucket_fqns)
        ]
        flex_shard(self.module, buckets=self.buckets)

        for name, param in self.module.named_parameters():
            for attr_name, attr_value in custom_attrs.get(name, {}).items():
                setattr(param, attr_name, attr_value)
            param.is_flex_shard_param = True

        log_single_rank(
            logger,
            logging.INFO,
            f"FlexShard: {len(self.buckets)} buckets over {self.device_mesh.size()} ranks, "
            f"reshard_after_forward={ddp_config.reshard_after_forward}, "
            f"local params={sum(p.numel() for p in self.module.parameters())}",
        )

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
