# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TransformerEngine blockwise FP8 weights for FlexShard's FP8 parameter all-gather.

FlexShard keeps each weight's local shard in bf16. Its blockwise FP8 placement quantizes each
rank's 128-row block rows before the all-gather and builds the gathered weight from the FP8 data
and scales. With TransformerEngine's own weight quantizer for the blockwise recipe, the gathered
bytes and scales equal TransformerEngine quantizing the full bf16 weight, since a 128 x 128 block
never straddles ranks and the scales are padded only along their columns. TransformerEngine's
layers then use the gathered ``Float8BlockwiseQTensor`` as is, instead of quantizing the weight
themselves.

flex_shard's placement (``Fp8BucketedBlockShard``) quantizes with torchao and hard-codes torchao's
scale layout, one fp32 scale per block. ``TEBlockwiseFp8Shard`` subclasses it to quantize with
TransformerEngine instead, whose scales are padded to a multiple of 4 per block row.
"""

from __future__ import annotations

from typing import Tuple

import torch
from flex_shard.custom_placements.fp8_bucketed_block_shard import (
    _VEC_ALIGN_BYTES,
    _VEC_COPY_DTYPE,
    _VEC_COPY_NBYTES,
    Fp8BucketedBlockShard,
    _align_up,
    _cat_byte_regions_from_flat_storage,
    _cat_byte_regions_from_rank_rows,
    _ceil_div,
    _is_vec_copy_viewable,
)
from flex_shard.flex_shard.placement_contract import PlacementUnshardResult
from flex_shard.flex_shard.utils import _record_copy_out_if_eager

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.quantization import get_fp8_te_dtype
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

BLOCK_SIZE = 128


def is_blockwise_fp8_weight(module: torch.nn.Module, name: str, param: torch.nn.Parameter) -> bool:
    """Whether ``module.<name>`` is a weight TransformerEngine quantizes with 128 x 128 blocks:
    the 2D weights of its linear layers (``weight``, or ``weight<i>`` of a grouped linear), with
    both dims multiples of 128."""
    return (
        isinstance(module, TransformerEngineBaseModule)
        and (name == "weight" or (name.startswith("weight") and name[len("weight") :].isdigit()))
        and param.dim() == 2
        and param.shape[0] % BLOCK_SIZE == 0
        and param.shape[1] % BLOCK_SIZE == 0
    )


class TEBlockwiseFp8Weights:
    """TransformerEngine's blockwise recipe (``Float8BlockScaling``) for ``TEBlockwiseFp8Shard``:
    its scale layout, its quantization, and flex_shard's ``BlockwiseFp8WeightFactory``.

    ``quantize`` uses the recipe's weight quantizer, row-wise only, on a rank's block rows. The
    factory builds a ``Float8BlockwiseQTensor`` over the gathered buffers, without column-wise
    data: TransformerEngine derives it from the row-wise data when backward needs it, and
    ``drop_columnwise`` frees it once the weights change.
    """

    def __init__(self, recipe) -> None:
        qparams = recipe.fp8_quant_fwd_weight
        kwargs = dict(
            fp8_dtype=get_fp8_te_dtype(recipe, fprop_tensor=True),
            amax_epsilon=qparams.amax_epsilon,
            force_pow_2_scales=qparams.power_2_scale,
            block_scaling_dim=recipe.w_block_scaling_dim,
        )
        assert kwargs["block_scaling_dim"] == 2, (
            "FlexShard's FP8 parameter all-gather needs 2D (128 x 128) weight blocks, got "
            f"block_scaling_dim={kwargs['block_scaling_dim']}"
        )
        self._rowwise_quantizer = Float8BlockQuantizer(rowwise=True, columnwise=False, **kwargs)
        self.weight_quantizer = Float8BlockQuantizer(rowwise=True, columnwise=True, **kwargs)
        # A stable callable: placements compare weight factories by identity.
        self.weight_factory = self._weight

    def scale_cols(self, in_dim: int, block_size: int) -> int:
        """Scales per block row: TransformerEngine pads them to a multiple of 4."""
        assert block_size == BLOCK_SIZE, f"expected {BLOCK_SIZE}-row blocks, got {block_size}"
        return self._rowwise_quantizer.get_scale_shape((block_size, in_dim), columnwise=False)[1]

    def quantize(
        self, weight: torch.Tensor, block_size: int, fp8_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a rank's block rows as TransformerEngine quantizes the whole weight."""
        assert block_size == BLOCK_SIZE, f"expected {BLOCK_SIZE}-row blocks, got {block_size}"
        quantized = self._rowwise_quantizer(weight)
        return quantized._rowwise_data, quantized._rowwise_scale_inv

    def _weight(
        self,
        fp8_data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        return Float8BlockwiseQTensor(
            shape=fp8_data.shape,
            dtype=orig_dtype,
            fp8_dtype=self.weight_quantizer.dtype,
            rowwise_data=fp8_data.view(torch.uint8),
            rowwise_scale_inv=recip_scale,
            columnwise_data=None,
            columnwise_scale_inv=None,
            quantizer=self.weight_quantizer,
            is_2D_scaled=True,
            requires_grad=requires_grad,
        )


class TEBlockwiseFp8Shard(Fp8BucketedBlockShard):
    """flex_shard's blockwise FP8 placement, quantizing with TransformerEngine.

    flex_shard hard-codes torchao's scale layout in three private methods; the copies below change
    only the scales per block row (``TEBlockwiseFp8Weights.scale_cols``). They follow flex_shard's
    internals (main at c316223), so re-sync them when flex_shard changes those methods. Inside a
    ``MixedBucketPlacement``, the mixed bucket lays out, packs and unpacks the FP8 weights through
    this placement instance (its ``_collective_placement``), so the overrides apply there too.
    """

    def __init__(self, *, world_size: int, weights: TEBlockwiseFp8Weights) -> None:
        super().__init__(
            world_size=world_size, weight_factory=weights.weight_factory, block_size=BLOCK_SIZE
        )
        self._te = weights

    def _quantize_local_weight(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._te.quantize(tensor, self.block_size, self.fp8_dtype)

    def _build_block_row_units(self, params: list[_BlockRowParam]) -> list[_BlockRowUnit]:
        fp8_itemsize = torch.empty((), dtype=self.fp8_dtype).element_size()
        scale_itemsize = torch.float32.itemsize
        units: list[Fp8BucketedBlockShard._BlockRowUnit] = []
        for param in params:
            out_dim, in_dim = param.global_shape
            num_col_blocks = self._te.scale_cols(in_dim, self.block_size)
            for row_start in range(0, out_dim, self.block_size):
                row_end = min(row_start + self.block_size, out_dim)
                dense_numel = (row_end - row_start) * in_dim
                units.append(
                    Fp8BucketedBlockShard._BlockRowUnit(
                        fqn=param.fqn,
                        row_start=row_start,
                        row_end=row_end,
                        dense_numel=dense_numel,
                        cost_bytes=dense_numel * fp8_itemsize + num_col_blocks * scale_itemsize,
                    )
                )
        return units

    def _build_fp8_metadata(
        self, params: list[_BlockRowParam], units: list[_BlockRowUnit], unit_cuts: tuple[int, ...]
    ) -> _Fp8BucketMetadata:
        params_by_fqn = {param.fqn: param for param in params}
        param_indices_by_fqn = {param.fqn: index for index, param in enumerate(params)}
        chunks: list[Fp8BucketedBlockShard._Fp8Chunk] = []
        rank_pack_plans: list[Fp8BucketedBlockShard._Fp8RankPackPlan] = []
        rank_fp8_numels: list[int] = []
        rank_scale_numels: list[int] = []
        for rank, (unit_start, unit_end) in enumerate(
            zip(unit_cuts[:-1], unit_cuts[1:], strict=True)
        ):
            rank_dense_offset = 0
            fp8_offset = 0
            scale_offset = 0
            rank_units = units[unit_start:unit_end]
            rank_pack_chunks: list[Fp8BucketedBlockShard._Fp8PackChunk] = []
            idx = 0
            while idx < len(rank_units):
                first = rank_units[idx]
                last = first
                idx += 1
                while idx < len(rank_units) and rank_units[idx].fqn == first.fqn:
                    last = rank_units[idx]
                    idx += 1
                param = params_by_fqn[first.fqn]
                _, in_dim = param.global_shape
                row_start = first.row_start
                row_end = last.row_end
                scale_row_start = row_start // self.block_size
                scale_row_end = _ceil_div(row_end, self.block_size)
                chunk_dense_numel = (row_end - row_start) * in_dim
                scale_numel = (scale_row_end - scale_row_start) * self._te.scale_cols(
                    in_dim, self.block_size
                )
                chunks.append(
                    Fp8BucketedBlockShard._Fp8Chunk(
                        rank=rank,
                        fqn=param.fqn,
                        row_start=row_start,
                        row_end=row_end,
                        scale_row_start=scale_row_start,
                        scale_row_end=scale_row_end,
                        global_dense_offset=param.param_offset + row_start * in_dim,
                        rank_dense_offset=rank_dense_offset,
                        fp8_offset=fp8_offset,
                        scale_offset=scale_offset,
                    )
                )
                rank_pack_chunks.append(
                    Fp8BucketedBlockShard._Fp8PackChunk(
                        param_index=param_indices_by_fqn[param.fqn],
                        fqn=param.fqn,
                        local_shape=torch.Size((row_end - row_start, in_dim)),
                        fp8_numel=chunk_dense_numel,
                        scale_numel=scale_numel,
                    )
                )
                rank_dense_offset += chunk_dense_numel
                fp8_offset += chunk_dense_numel
                scale_offset += scale_numel
            rank_fp8_numels.append(fp8_offset)
            rank_scale_numels.append(scale_offset)
            rank_pack_plans.append(
                Fp8BucketedBlockShard._Fp8RankPackPlan(
                    chunks=tuple(rank_pack_chunks),
                    fp8_numel=fp8_offset,
                    scale_byte_offset=_align_up(fp8_offset, _VEC_ALIGN_BYTES),
                    packed_nbytes=_align_up(fp8_offset, _VEC_ALIGN_BYTES)
                    + scale_offset * torch.float32.itemsize,
                    fp8_chunk_numels=tuple(chunk.fp8_numel for chunk in rank_pack_chunks),
                    scale_chunk_nbytes=tuple(
                        chunk.scale_numel * torch.float32.itemsize for chunk in rank_pack_chunks
                    ),
                )
            )
        scale_itemsize = torch.float32.itemsize
        rank_scale_byte_offsets = tuple(
            _align_up(fp8_numel, _VEC_ALIGN_BYTES) for fp8_numel in rank_fp8_numels
        )
        rank_packed_nbytes = tuple(
            scale_byte_offset + scale_numel * scale_itemsize
            for scale_byte_offset, scale_numel in zip(
                rank_scale_byte_offsets, rank_scale_numels, strict=True
            )
        )
        return Fp8BucketedBlockShard._Fp8BucketMetadata(
            chunks=tuple(chunks),
            rank_pack_plans=tuple(rank_pack_plans),
            rank_fp8_numels=tuple(rank_fp8_numels),
            rank_scale_numels=tuple(rank_scale_numels),
            rank_scale_byte_offsets=rank_scale_byte_offsets,
            rank_packed_nbytes=rank_packed_nbytes,
            packed_send_nbytes_per_rank=_align_up(
                max(rank_packed_nbytes, default=0), _VEC_ALIGN_BYTES
            ),
        )

    def _finish_fp8_unshard_from_rank_rows(
        self, prepared: PlacementPreparedUnshard, rank_rows: torch.Tensor, state: _Fp8UnshardState
    ) -> PlacementUnshardResult:
        metadata = state.metadata
        fp8_nbytes = sum(metadata.rank_fp8_numels)
        scale_nbytes = sum(metadata.rank_scale_numels) * torch.float32.itemsize
        scale_byte_offset = _align_up(fp8_nbytes, _VEC_ALIGN_BYTES)
        # Persistent unshards keep ``compact`` (fp8 data + scales) as the
        # storage backing the returned BlockwiseFp8Weights.
        compact = (
            prepared.persistent_buffers[0]
            if prepared.persistent_buffers is not None
            else torch.empty(
                scale_byte_offset + scale_nbytes, dtype=torch.uint8, device=rank_rows.device
            )
        )
        fp8_bytes = compact[:fp8_nbytes]
        scale_bytes = compact[scale_byte_offset:]
        row_stride_bytes = rank_rows.stride(0)
        fp8_storage_regions = tuple(
            (rank * row_stride_bytes, rank_fp8_numel)
            for rank, rank_fp8_numel in enumerate(metadata.rank_fp8_numels)
        )
        scale_storage_regions = tuple(
            (rank * row_stride_bytes + scale_byte_start, rank_scale_numel * torch.float32.itemsize)
            for rank, (rank_scale_numel, scale_byte_start) in enumerate(
                zip(metadata.rank_scale_numels, metadata.rank_scale_byte_offsets, strict=True)
            )
        )

        with _record_copy_out_if_eager():
            # Rank cuts partition one globally ordered block-row stream. Joining
            # each rank's valid regions therefore restores parameter order without
            # scattering every chunk separately. View the physical row storage once,
            # including any mixed-bucket padding between rows, so cat inputs do not
            # need intermediate per-rank views. Retain byte cats for unaligned or
            # odd inputs.
            use_vec_copies = _is_vec_copy_viewable(rank_rows) and _is_vec_copy_viewable(compact)
            if use_vec_copies:
                rank_storage_bytes = rank_rows.as_strided(
                    ((rank_rows.size(0) - 1) * row_stride_bytes + rank_rows.size(1),), (1,)
                )
                rank_storage_copy_units = rank_storage_bytes.view(_VEC_COPY_DTYPE)
                compact_copy_units = compact.view(_VEC_COPY_DTYPE)
                if all(
                    rank_fp8_numel % _VEC_COPY_NBYTES == 0
                    for rank_fp8_numel in metadata.rank_fp8_numels
                ):
                    _cat_byte_regions_from_flat_storage(
                        fp8_storage_regions,
                        rank_storage_copy_units,
                        compact_copy_units[: fp8_nbytes // _VEC_COPY_NBYTES],
                        copy_unit_nbytes=_VEC_COPY_NBYTES,
                    )
                else:
                    _cat_byte_regions_from_flat_storage(
                        fp8_storage_regions, rank_storage_bytes, fp8_bytes
                    )
                _cat_byte_regions_from_flat_storage(
                    scale_storage_regions,
                    rank_storage_copy_units,
                    compact_copy_units[scale_byte_offset // _VEC_COPY_NBYTES :],
                    copy_unit_nbytes=_VEC_COPY_NBYTES,
                )
            else:
                rank_byte_rows = rank_rows.unbind(0)
                _cat_byte_regions_from_rank_rows(
                    fp8_storage_regions,
                    rank_byte_rows,
                    fp8_bytes,
                    row_stride_bytes=row_stride_bytes,
                )
                _cat_byte_regions_from_rank_rows(
                    scale_storage_regions,
                    rank_byte_rows,
                    scale_bytes,
                    row_stride_bytes=row_stride_bytes,
                )

        fp8_flat = fp8_bytes.view(self.fp8_dtype)
        scale_flat = scale_bytes.view(torch.float32)
        fp8_offset = 0
        scale_offset = 0
        full_params: list[torch.Tensor] = []
        for info in state.infos:
            fp8_numel = info.global_numel
            scale_shape = (
                _ceil_div(info.global_shape[0], self.block_size),
                self._te.scale_cols(info.global_shape[1], self.block_size),
            )
            scale_numel = scale_shape[0] * scale_shape[1]
            fp8_data = fp8_flat[fp8_offset : fp8_offset + fp8_numel].view(info.global_shape)
            recip_scale = scale_flat[scale_offset : scale_offset + scale_numel].view(scale_shape)
            full_params.append(
                self._make_blockwise_fp8_weight(
                    fp8_data,
                    recip_scale,
                    orig_dtype=info.unsharded_dtype,
                    requires_grad=info.requires_grad,
                )
            )
            fp8_offset += fp8_numel
            scale_offset += scale_numel
        if fp8_offset != fp8_flat.numel() or scale_offset != scale_flat.numel():
            raise AssertionError("Compacted FP8 bucket metadata is inconsistent.")
        return PlacementUnshardResult(
            full_params=full_params, persistent_buffers=[compact] if prepared.persistent else []
        )


def drop_columnwise(param: torch.Tensor) -> None:
    """Free the column-wise data TransformerEngine derived from a gathered FP8 weight for
    backward. Its row-wise data views FlexShard's gathered buffer, which the next all-gather
    refills with the updated weights, so the derived copy must not outlive them."""
    if HAVE_TE and isinstance(param, Float8BlockwiseQTensor) and param._columnwise_data is not None:
        param.update_usage(rowwise_usage=True, columnwise_usage=False)
