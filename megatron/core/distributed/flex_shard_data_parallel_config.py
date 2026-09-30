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
