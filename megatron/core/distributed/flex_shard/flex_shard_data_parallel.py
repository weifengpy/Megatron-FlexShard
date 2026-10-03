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

    With tied embeddings, the model fetches the output weight before it calls the output layer
    and passes it in at call time, so FlexShard cannot see that use from parameter names. On the
    stage that holds the embedding (the only stage without pipeline parallelism, the MTP stage
    with it), the output layer registers no weight and reuses the embedding's; on a last stage
    without it, the weight is the output layer's own copy. The final norm's parameters join the
    bucket holding that weight, as torchtitan groups [tok_embeddings, norm, lm_head] for FSDP2:
    their deepest common module is the model root, so the bucket's hooks gather the weight before
    the model fetches it and reduce-scatter it after every use. The bucket skips
    reshard-after-forward, which at the root would free the weight right before backward
    re-gathers it.

    With expert parallelism, expert parameters (``allreduce=False``) exist only on their EP rank
    and are replicated over the expert data-parallel group, so each MoE layer's experts get
    their own bucket on that group, hooked on the experts module, as Megatron DDP keeps them in
    separate buffers. Each expert's gradient already sums the tokens its EP peers routed to it,
    so its bucket divides by the dense data-parallel size, not its own group's (Megatron DDP's
    expert gradient scaling, FSDP2's gradient divide factor). With per-token loss, every bucket
    sums instead, and finalize_model_grads divides by the global token count.

    With pipeline parallelism, each model chunk (one per virtual pipeline stage) is its own
    FlexShardDataParallel. Megatron's schedules run each microbatch's backward separately and
    re-enable sync before each chunk's last microbatch backward, which reduce-scatters. Tied
    embedding and output weights on the first and last stages are separate copies, which
    finalize_model_grads all-reduces over the embedding group on the local shards; both stages
    shard them identically.

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
        self.expert_process_group = None
        self.expert_device_mesh = None
        if any(_is_expert_param(param) for param in self.module.parameters()):
            self.expert_process_group = getattr(pg_collection, "expt_dp", None)
            if self.expert_process_group is None:
                # Migration fallback for callers that do not pass process groups.
                self.expert_process_group = parallel_state.get_expert_data_parallel_group()
            self.expert_device_mesh = DeviceMesh.from_group(
                self.expert_process_group, "cuda", mesh_dim_names=("edp",)
            )

        # FlexShard replaces each parameter with a local-shard tensor, dropping the Megatron
        # attributes (tensor_model_parallel, allreduce, ...) the optimizer relies on.
        custom_attrs = {
            name: dict(vars(param)) for name, param in self.module.named_parameters()
        }

        mp_policy = MixedPrecisionPolicy(
            reduce_dtype=torch.float32 if ddp_config.grad_reduce_in_fp32 else None
        )
        tied = self._find_tied_output_weight()
        bucket_fqns = self._build_bucket_fqns(tied)
        tied_bucket = None
        if tied is not None:
            weight_fqn, output_layer_path = tied
            tied_bucket = next(i for i, fqns in enumerate(bucket_fqns) if weight_fqn in fqns)
            anchor = _common_module_path(bucket_fqns[tied_bucket])
            assert anchor == "" or output_layer_path.startswith(anchor + "."), (
                "FlexShard with tied embeddings needs the embedding bucket's hooks on a module "
                f"that also runs the output layer, got {anchor!r} for {output_layer_path!r}."
            )
        fused_fqns = self._fused_wgrad_fqns(tied) if config.gradient_accumulation_fusion else set()
        uses_main_grad = [any(fqn in fused_fqns for fqn in fqns) for fqns in bucket_fqns]
        # Passed only where needed, so FlexShard without fusion still works with a flex_shard
        # that predates the hooks.
        main_grad_hooks = dict(
            pre_backward_hook=self._alias_main_grads, post_reduce_hook=self._drop_main_grads
        )
        params = dict(self.module.named_parameters())
        is_expert = [_is_expert_param(params[fqns[0]]) for fqns in bucket_fqns]
        self.buckets = [
            BucketSpec(
                fqns,
                placement_fn=per_param_placements,
                mesh=self.expert_device_mesh if is_expert[i] else self.device_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=(
                    ddp_config.reshard_after_forward
                    and i < len(bucket_fqns) - 1
                    and i != tied_bucket
                ),
                **self._gradient_reduction(config, is_expert[i]),
                **(main_grad_hooks if uses_main_grad[i] else {}),
            )
            for i, fqns in enumerate(bucket_fqns)
        ]
        if ddp_config.grad_reduce_in_fp32:
            # FlexShard stores each local shard's gradient in the parameter's grad_dtype
            # (flex_shard#23), so bf16 parameters get fp32 shard gradients, which the
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
            f"FlexShard: {len(self.buckets) - sum(is_expert)} buckets over "
            f"{self.device_mesh.size()} ranks, {sum(is_expert)} expert buckets over "
            f"{self.expert_device_mesh.size() if self.expert_device_mesh else 0} ranks, "
            f"reshard_after_forward={ddp_config.reshard_after_forward}, "
            f"no_sync={ddp_config.no_sync}, "
            f"reshard_after_backward={ddp_config.reshard_after_backward}, "
            f"main_grad buckets={sum(uses_main_grad)}, "
            f"tied embeddings={tied is not None}, "
            f"local params={sum(p.numel() for p in self.module.parameters())}",
        )

    def _gradient_reduction(self, config: TransformerConfig, is_expert: bool) -> Dict:
        """BucketSpec reduction arguments that scale gradients as Megatron DDP does."""
        if config.calculate_per_token_loss:
            # finalize_model_grads divides every gradient by the global token count.
            return dict(gradient_reduce_op=torch.distributed.ReduceOp.SUM)
        if is_expert and self.expert_device_mesh.size() != self.device_mesh.size():
            # An expert's gradient already sums the tokens its EP peers routed to it, so the
            # sum over the expert data-parallel group is divided by the dense data-parallel
            # size. Passed only where needed, so FlexShard without expert parallelism still
            # works with a flex_shard that predates gradient_divide_factor.
            return dict(gradient_divide_factor=self.device_mesh.size())
        return {}

    def _find_tied_output_weight(self) -> Optional[Tuple[str, str]]:
        """``(weight FQN, output-layer path)`` if the output layer reuses the embedding weight."""
        for name, submodule in self.module.named_modules():
            # With tied weights, the model fetches the output weight before it calls the output
            # layer and passes it in: the embedding's weight on the stage that holds the
            # embedding (the only stage without pipeline parallelism, or the MTP stage), else
            # the output layer's own copy. Either way its bucket must be gathered by then.
            if not (
                getattr(submodule, "share_embeddings_and_output_weights", False)
                and getattr(submodule, "post_process", False)
            ):
                continue
            weight = submodule.shared_embedding_or_output_weight()
            weight_fqn = next(
                fqn for fqn, param in self.module.named_parameters() if param is weight
            )
            return weight_fqn, f"{name}.output_layer" if name else "output_layer"
        return None

    def _fused_wgrad_fqns(self, tied: Optional[Tuple[str, str]]) -> Set[str]:
        """FQNs of parameters whose modules add weight gradients into ``main_grad``."""
        fqns = {
            f"{name}.{param_name}" if name else param_name
            for name, submodule in self.module.named_modules()
            if getattr(submodule, "fuse_wgrad_accumulation", False)
            or getattr(submodule, "gradient_accumulation_fusion", False)
            for param_name, _ in submodule.named_parameters(recurse=False)
        }
        if tied is not None:
            weight_fqn, output_layer_path = tied
            output_layer = self.module.get_submodule(output_layer_path)
            if getattr(output_layer, "gradient_accumulation_fusion", False):
                # The output layer adds the tied weight's gradient into main_grad.
                fqns.add(weight_fqn)
        return fqns

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

    def _build_bucket_fqns(self, tied: Optional[Tuple[str, str]]) -> List[List[str]]:
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

        if tied is not None:
            # The final norm joins the bucket holding the tied weight (the embedding's, or the
            # output layer's own copy), so the bucket's common module, where FlexShard hooks it,
            # is the model root that fetches the weight and runs the output layer.
            weight_fqn, output_layer_path = tied
            norm_prefix = output_layer_path[: -len("output_layer")] + "decoder.final_layernorm."
            tied_fqns = next(fqns for fqns in buckets if weight_fqn in fqns)
            norms = [fqns for fqns in buckets if fqns[0].startswith(norm_prefix)]
            if not norms:
                raise ValueError(
                    "FlexShard with tied embeddings groups the final norm with the tied weight, "
                    f"but found no parameters under {norm_prefix!r}."
                )
            for fqns in norms:
                tied_fqns.extend(fqns)
                buckets.remove(fqns)

        # Expert parameters shard over the expert data-parallel group, so an MoE layer's
        # experts get their own bucket after its other parameters; FlexShard also requires one
        # parameter dtype per bucket.
        split_buckets = []
        for fqns in buckets:
            groups: Dict[Tuple[bool, torch.dtype], List[str]] = {}
            for fqn in fqns:
                key = (_is_expert_param(params[fqn]), params[fqn].dtype)
                groups.setdefault(key, []).append(fqn)
            split_buckets.extend(groups.values())
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


def _is_expert_param(param: torch.nn.Parameter) -> bool:
    """Whether ``param`` uses the expert topology (EP/ETP/expert data parallelism)."""
    return not getattr(param, "allreduce", True)


def _common_module_path(fqns: List[str]) -> str:
    """Deepest module path that contains every parameter in ``fqns``."""
    common: List[str] = []
    for parts in zip(*(fqn.split(".")[:-1] for fqn in fqns)):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return ".".join(common)
