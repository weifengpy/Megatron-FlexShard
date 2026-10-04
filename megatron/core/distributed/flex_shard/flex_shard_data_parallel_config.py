# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig


@dataclass
class FlexShardDataParallelConfig(DistributedDataParallelConfig):
    """Configuration for FlexShardDataParallel."""

    reshard_after_forward: bool = True
    """Free each bucket's unsharded parameters after forward and all-gather them again in
    backward (ZeRO-3). If False, keep them until backward (ZeRO-2). The last bucket in forward
    order never reshards, since its backward runs immediately.
    """

    no_sync: bool = False
    """Reduce-scatter gradients only in the last microbatch's backward. Earlier microbatches
    run under no_sync_func and accumulate full gradients (in fp32 with grad_reduce_in_fp32),
    which costs one full gradient copy of memory.
    """

    reshard_after_backward: bool = True
    """Free each bucket's unsharded parameters after a backward without gradient sync. If
    False, keep them for the next microbatch, which then skips the all-gather: with no_sync
    and without reshard_after_forward, one all-gather and one reduce-scatter per bucket per
    step. The last microbatch's backward always reshards.
    """

    own_matrices: bool = False
    """Store each bucket that holds a Muon matrix with whole-parameter owners (flex_shard's
    BucketedOwned): each parameter lives on one rank, which receives its whole gradient, so
    Muon orthogonalizes complete matrices locally, without optimizer communication. The
    embedding and output buckets keep row shards.
    """
