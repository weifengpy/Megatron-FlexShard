# Plan: tensor parallelism with FlexShard in Megatron

## Goal

Run FlexShard without reshard-after-forward at TP × DP, and compare it with Megatron DDP + distributed optimizer at the same TP × DP. Today `--use-flex-shard` rejects `--tensor-model-parallel-size > 1`. TP is the usual setup for dense models from about 8B up, so larger comparisons need it.

## How it fits together

With TP, each rank holds its tensor-parallel slice of each weight, and Megatron's layers do the TP communication themselves:
- ColumnParallel and RowParallel linears, including the TransformerEngine ones;
- sequence-parallel all-gathers and reduce-scatters.

FlexShard only sees those slices as ordinary parameters. It shards them over the rank's data-parallel group, which `FlexShardDataParallel` already takes from `pg_collection.dp_cp`, and that group excludes the TP ranks. The two layers don't touch: FlexShard all-gathers and reduce-scatters each TP slice within its data-parallel group, and the model's TP communication is unchanged. So the code change should be small, and most of the work is verification.

FlexShard's Megatron integration lives in its own folder, `megatron/core/distributed/flex_shard/` (moved there in PR #5, as Megatron-FSDP lives in `megatron/core/distributed/fsdp/`). The TP changes, including the layout adapter in step 8, go there.

## What should already work

**Gradient norm and clipping.**
- For FlexShard, the optimizer reduces gradient statistics over WORLD (`megatron/core/optimizer/__init__.py`), since each element of a local-shard gradient lives on exactly one data-parallel rank.
- With TP, parameters replicated across TP ranks (e.g. layer norms) must still be counted once. Megatron's existing filter does that: `get_main_grads_for_grad_norm` in `megatron/core/optimizer/optimizer.py` calls `param_is_not_tensor_parallel_duplicate`, which counts them only on TP rank 0.
- That filter reads each parameter's `tensor_model_parallel` attribute. FlexShard's wrapper copies Megatron's parameter attributes onto its local shards after `flex_shard()`.

**Sequence-parallel layer-norm gradients.**
- With `--sequence-parallel`, each TP rank computes only part of the layer-norm gradients.
- `_allreduce_non_tensor_model_parallel_grads` in `megatron/core/distributed/finalize_model_grads.py` sums them across TP ranks. Without a `main_grad`, which is FlexShard's case, it reduces `param.grad`, here FlexShard's local-shard gradient.
- That is correct: every TP rank holds the same layer-norm parameters and shards them the same way over its data-parallel group, so the shards line up element by element.
- It runs after backward, when FlexShard's reduce-scatters are done. Both reductions are sums or averages, so their order doesn't matter.

## Steps

1. **Lift the restriction.**
   - Remove the `--use-flex-shard` check in `validate_args` (`megatron/training/arguments.py`) that rejects `--tensor-model-parallel-size > 1`.
   - Check that the wrapper's attribute restore keeps `tensor_model_parallel`, `partition_dim`, `partition_stride` and `sequence_parallel` on FlexShard's local shards. The wrapper is `megatron/core/distributed/flex_shard/flex_shard_data_parallel.py`.
2. **Check gradient norm and clipping.** Iteration-1 grad norm must match Megatron at the same TP × DP. A double-counted replicated parameter would show up as a larger norm.
3. **Check sequence parallelism.** Test with and without `--sequence-parallel`, and with `--qk-layernorm`, whose layer-norm gradients take the same all-reduce.
4. **Correctness runs.**
   - Start on the tiny model, then shape S.
   - Use TP 2 × DP 2 on 4 GPUs, then TP 2 × DP 4 and TP 4 × DP 2 on 8 GPUs.
   - Use the same criteria as Phase A in the README: iteration-1 loss and grad norm, then loss curves within the spread between two Megatron runs.
   - Run with and without `--flex-shard-no-sync --flex-shard-no-reshard-after-backward`, since TP interacts with gradient accumulation.
5. **Fix what breaks.** Likely spots:
   - TransformerEngine's TP communication overlap (`--tp-comm-overlap`).
   - Any TP code path that expects `main_grad`.
   - Mixed gradient dtypes in the TP all-reduce, which flattens gradients together and needs one dtype. With `--accumulate-allreduce-grads-in-fp32`, FlexShard's local-shard gradients should all be fp32.
6. **Benchmark.** Shape L (8B) at TP 2 × DP 4 on 8 GPUs, with the README's Phase A performance protocol:
   - median time per iteration, as the min and median of interleaved repetitions;
   - peak allocated memory;
   - a one-step profile of each setup.
7. **README.** Document the TP design, and remove tensor parallelism from the list of rejected options.
8. **Later, with checkpointing: declare each TP slice's place in the full weight.** See [Layout adapter](#layout-adapter-step-8) below. Steps 1–7 don't depend on it.

## Layout adapter (step 8)

**Why training doesn't need it.** FlexShard reads a parameter's declared layout (`outer_layout`) only in its checkpoint code (`flex_shard/checkpoint.py`, through `compose_global_layouts`). That code works out where each FlexShard shard sits in the full tensor. Sharding, all-gather and reduce-scatter never use the layout. Checkpoint save/load needs it, and so does anything else that needs a shard's place in the full parameter, such as loading at a different TP or DP size.

**Design.**
- Add `megatron/core/distributed/flex_shard/layout.py`. `FlexShardDataParallel` calls it before `flex_shard()`, and it declares each parameter's `GlobalLayout` with flex_shard's `set_global_layout`.
- It builds each layout from Megatron's own checkpoint metadata, `model.sharded_state_dict()`. Megatron already computes there each parameter's full shape and offsets under TP, EP and PP, including special cases:
  - SwiGLU's fused fc1 stores `[gate; up]` per rank, so each rank holds two separate regions of the full tensor.
  - Padded vocabularies.
- `GlobalLayout` follows DCP's checkpoint layout format and accepts several regions per local tensor, so each `sharded_state_dict()` entry maps to one or more regions.
- Checkpoints then use the same keys and full shapes as Megatron's own, so a Megatron checkpoint could load into FlexShard and back.

**Alternatives not chosen.**
- **Megatron's TP attributes** (`tensor_model_parallel`, `partition_dim`, `partition_stride`) plus the TP rank and size. This is simpler and covers ordinary column- and row-parallel weights, but it may miss the special layouts above.
- **flex_shard's existing adapters.**
  - `dtensor_to_global_layout` needs DTensor parameters. Megatron's TP parameters aren't DTensors, and converting the fused layouts would need DTensor's strided-shard placement, which the adapter rejects.
  - `spmd_types_to_global_layout` needs spmd_types layout descriptions, which Megatron doesn't produce.

**Why the adapter lives in Megatron, not in flex_shard's `layout_adapters`:**
- It reads Megatron internals that change across Megatron versions. Next to `FlexShardDataParallel`, it changes and gets tested along with them.
- flex_shard's CI has no Megatron, so in flex_shard it would effectively go untested.
- Framework integration code lives in the framework. torchtitan's FlexShard code also lives in torchtitan.

flex_shard already provides the interface: `set_global_layout` and multi-region `GlobalLayout`. On the flex_shard side, add one Megatron-free test: a local tensor that declares several regions composes correctly with FlexShard's own sharding in `compose_global_layouts`.

**Verification.**
- Collect every rank's local shards with their declared regions, reassemble the full tensors, and compare them with a TP = 1 model initialized identically (`--use-cpu-initialization`).
- Then save a checkpoint at one TP × DP size and load it at another.

## Validation matrix

| Model | TP × DP | GPUs | Sequence parallel | Gradient accumulation | Check |
| --- | --- | --- | --- | --- | --- |
| Tiny (4 layers) | 2 × 2 | 4 | off, on | 1 and 4 microbatches, with and without no-sync | iteration-1 loss and grad norm, 30-iteration loss |
| Shape S | 2 × 2 | 4 | on | 2 microbatches, with no-sync | ~500-iteration loss within Megatron's run-to-run spread |
| Shape S | 2 × 4, 4 × 2 | 8 | on | 2 microbatches, with no-sync | same |
| Shape L (8B) | 2 × 4 | 8 | on | 1, 2 and 8 microbatches | loss parity, then time and memory |

## Risks

- **TransformerEngine's TP communication overlap** (`--tp-comm-overlap`) is the one real unknown. It needs to be run to see whether it composes with FlexShard's parameter swap.
- **Code paths that assume Megatron DDP's `main_grad` buffers** under TP, beyond the layer-norm all-reduce checked above.

## Effort

Steps 1–7 probably take a day:
- about an hour to lift the check and fix small issues;
- the rest on the correctness runs, mainly the 8-GPU ones.

Step 8, the layout adapter, is separate work that ships with checkpoint support.
