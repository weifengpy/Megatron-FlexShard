# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron ``torch_dist`` checkpoints of FlexShard's local shards.

Megatron's modules describe each parameter as one piece of a global tensor (``ShardedTensor``),
computing its key, global shape and offsets from the parameter's shape and the TP/PP/EP ranks,
and treating data-parallel ranks as replicas. Under FlexShard a module's parameter is only this
rank's local shard, and flex_shard describes where that shard sits in the full parameter with
DCP's ``CheckpointableTensor`` layout: per chunk, a global offset, a local offset and a size
(``get_flex_shard_global_layouts``). So:

- The module's own ``sharded_state_dict()`` runs on meta stand-ins of the full parameters
  (``device="meta"``, no memory). It emits exactly what it emits for Megatron DDP: keys, global
  shapes, TP/PP/EP offsets, prepended layer and expert axes, replica ids and factories such as
  SwiGLU's gate/up split, with meta data.
- Each entry built from a stand-in becomes a ``ShardedTensorFactory`` over the local shard. Its
  build intersects Megatron's pieces with flex_shard's chunks and emits one ``ShardedTensor`` per
  overlap: a view of the shard at the piece's global offset plus the overlap's position, with
  ``axis_fragmentations=None`` (uneven pieces, which Megatron leaves to DCP to validate) and the
  data-parallel part of ``replica_id`` zeroed, since each data-parallel rank owns its chunks. Its
  merge writes loaded chunks back into a tensor of the shard's shape for ``load_state_dict``.
- The build slices whatever tensor it is given, so Megatron's optimizer, which replaces a
  factory's data with each optimizer state of the parameter (``make_sharded_optimizer_tensor``),
  reuses it for the fp32 main parameters and Adam moments, which have the shard's shape.

The checkpoint thus has Megatron DDP's keys and global shapes, and the two load each other's.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...dist_checkpointing.dict_utils import dict_list_map_inplace, nested_values
from ...dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
)

try:
    from flex_shard import get_flex_shard_global_layouts

    HAVE_FLEX_SHARD = True
except ImportError:
    HAVE_FLEX_SHARD = False

Region = Tuple[Tuple[int, ...], Tuple[int, ...]]  # (offsets, sizes) in the full parameter


def _region_in(view: torch.Tensor, base: torch.Tensor) -> Region:
    """Where a basic-slicing view of ``base`` (e.g. a ``torch.chunk``) sits in it."""
    assert view.stride() == base.stride(), (
        f"Expected a slice of the full parameter, got strides {view.stride()} vs {base.stride()}"
    )
    offset = view.storage_offset() - base.storage_offset()
    starts = []
    for stride in base.stride():
        starts.append(offset // stride)
        offset %= stride
    return tuple(starts), tuple(view.shape)


def _owned(replica_id: ReplicaId) -> ReplicaId:
    """Megatron's replica id with the data-parallel part zeroed: each rank owns its chunks."""
    if isinstance(replica_id, int):
        return 0
    assert len(replica_id) == 3, f"Expected a (pp, tp, dp) replica id, got {replica_id}"
    return (*replica_id[:-1], 0)


class _ChunkPlan:
    """The overlaps of Megatron's pieces of one parameter with its local shard's chunks."""

    def __init__(self, regions: List[Region], layout: Any, local_shape: torch.Size):
        self.local_shape = local_shape
        # (piece index, start in the full parameter, size, start in the local shard)
        self.chunks: List[Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = []
        for index, (r_start, r_size) in enumerate(regions):
            for g_start, l_start, size in zip(
                layout.global_offsets, layout.local_offsets, layout.local_sizes, strict=True
            ):
                start = tuple(map(max, g_start, r_start))
                end = tuple(
                    min(g + s, r + t) for g, s, r, t in zip(g_start, size, r_start, r_size)
                )
                if any(e <= s for s, e in zip(start, end)):
                    continue
                self.chunks.append(
                    (
                        index,
                        start,
                        tuple(e - s for s, e in zip(start, end)),
                        tuple(l + s - g for l, s, g in zip(l_start, start, g_start)),
                    )
                )
        self.regions = regions

    @staticmethod
    def _slices(start: Tuple[int, ...], size: Tuple[int, ...]) -> Tuple[slice, ...]:
        return tuple(slice(b, b + s) for b, s in zip(start, size))

    def build(self, pieces: List[ShardedTensor], tensor: torch.Tensor) -> List[ShardedTensor]:
        """One ShardedTensor per overlap, viewing ``tensor`` (shaped like the local shard)."""
        assert tensor.shape == self.local_shape, (tensor.shape, self.local_shape)
        out = []
        for index, start, size, local_start in self.chunks:
            piece = pieces[index]
            prepend = piece.prepend_axis_num
            r_start = self.regions[index][0]
            data = tensor[self._slices(local_start, size)]
            global_offset = piece.global_offset[:prepend] + tuple(
                g + s - r for g, s, r in zip(piece.global_offset[prepend:], start, r_start)
            )
            out.append(
                dataclasses.replace(
                    piece,
                    data=data,
                    dtype=data.dtype,
                    local_shape=size,
                    global_offset=global_offset,
                    axis_fragmentations=None,
                    replica_id=_owned(piece.replica_id),
                )
            )
        return out

    def merge(self, loaded: List[torch.Tensor], like: Optional[torch.Tensor]) -> torch.Tensor:
        """The local shard rebuilt from the loaded chunks, in ``build``'s order."""
        assert len(loaded) == len(self.chunks), (len(loaded), len(self.chunks))
        ref = loaded[0] if loaded else like
        out = torch.zeros(self.local_shape, dtype=ref.dtype, device=ref.device)
        for (_, _, size, local_start), chunk in zip(self.chunks, loaded):
            out[self._slices(local_start, size)].copy_(chunk)
        return out


def _local_shard_factory(
    entry: Any, stand_in: torch.Tensor, local: torch.nn.Parameter, layout: Any
) -> ShardedTensorFactory:
    """A factory over ``local`` that emits ``entry``'s pieces cut down to the local chunks."""
    if isinstance(entry, ShardedTensorFactory):
        # E.g. SwiGLU's gate/up split: Megatron's own build on the stand-in gives the pieces.
        def megatron_pieces(key, replica_id):
            built = entry.build_fn(key, stand_in, replica_id, None)
            return list(nested_values(built))

    else:

        def megatron_pieces(key, replica_id):
            return [dataclasses.replace(entry, key=key, replica_id=replica_id)]

    pieces = megatron_pieces(entry.key, entry.replica_id)
    regions = [_region_in(piece.data, stand_in) for piece in pieces]
    plan = _ChunkPlan(regions, layout, local.shape)
    last_built = {}

    def build_fn(key, tensor, replica_id, flattened_range):
        assert flattened_range is None
        last_built["tensor"] = tensor
        return plan.build(megatron_pieces(key, replica_id), tensor)

    def merge_fn(loaded):
        return plan.merge(loaded, last_built.get("tensor"))

    return ShardedTensorFactory(
        key=entry.key, data=local, build_fn=build_fn, merge_fn=merge_fn, replica_id=entry.replica_id
    )


def flex_shard_sharded_state_dict(
    module: torch.nn.Module,
    prefix: str = '',
    sharded_offsets: Tuple = (),
    metadata: Optional[Dict] = None,
) -> ShardedStateDict:
    """``module.sharded_state_dict()`` for a module FlexShard has sharded, in Megatron DDP's
    format (see the module docstring)."""
    # FlexShard's state_dict pre-hooks put the local shards back on the modules, so nothing
    # swaps them in over the stand-ins below.
    module.state_dict(keep_vars=True)
    layouts = get_flex_shard_global_layouts(module)
    stand_ins: Dict[int, torch.nn.Parameter] = {}
    by_stand_in: Dict[int, Tuple[torch.nn.Parameter, torch.nn.Parameter, Any]] = {}
    slots = []
    for fqn, layout in layouts.items():
        owner_path, _, name = fqn.rpartition('.')
        owner = module.get_submodule(owner_path)
        local = owner._parameters[name]
        stand_in = stand_ins.get(id(local))
        if stand_in is None:
            stand_in = torch.nn.Parameter(
                torch.empty(layout.global_shape, dtype=local.dtype, device="meta"),
                requires_grad=local.requires_grad,
            )
            for key, value in vars(local).items():
                setattr(stand_in, key, value)
            stand_ins[id(local)] = stand_in
            by_stand_in[id(stand_in)] = (stand_in, local, layout)
        slots.append((owner, name, local))
        owner._parameters[name] = stand_in
    try:
        sharded_state_dict = module.sharded_state_dict(prefix, sharded_offsets, metadata)
    finally:
        for owner, name, local in slots:
            owner._parameters[name] = local

    def convert(entry):
        if isinstance(entry, (ShardedTensor, ShardedTensorFactory)):
            found = by_stand_in.get(id(entry.data))
            if found is not None:
                return _local_shard_factory(entry, *found)
            assert entry.data is None or not entry.data.is_meta, (
                f"{entry.key} holds a stand-in that the adapter can't map back to its local shard"
            )
        return entry

    dict_list_map_inplace(convert, sharded_state_dict)
    return sharded_state_dict
