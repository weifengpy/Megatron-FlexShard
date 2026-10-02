# FlexShard data-parallel backend for Megatron-LM

This fork adds [FlexShard](https://github.com/meta-pytorch/flex_shard) as a Megatron data-parallel backend behind `--use-flex-shard`, alongside DDP, the distributed optimizer, torch FSDP2 and Megatron-FSDP. With the flag off, Megatron behaves exactly like upstream, so one `pretrain_gpt.py` command compares Megatron's DDP / DistributedOptimizer with FlexShard by flipping a flag.

Base: upstream NVIDIA/Megatron-LM `16251ac12` plus one commit, "Add FlexShard data-parallel backend (--use-flex-shard)".

## Usage

### Requirements

- PyTorch with CUDA and NCCL. Tested with a PyTorch 2.15 dev build on CUDA 13 (flex_shard declares `torch>=2.14,<2.15`, but its test suite passes on 2.15).
- `flex_shard` importable (`pip install --no-deps -e <flex_shard>` or `PYTHONPATH=<flex_shard>/src`). It also needs `torchao`, which is not declared: `flex_shard.custom_placements` imports the fp8 placement unconditionally.
- TransformerEngine layers (`--transformer-impl transformer_engine`) need flex_shard `main` at or after [#16](https://github.com/meta-pytorch/flex_shard/pull/16) (persistent unsharded parameters, which superseded drafts #13–#15). `--flex-shard-no-sync` and `--flex-shard-no-reshard-after-backward` need `main` at or after [#18](https://github.com/meta-pytorch/flex_shard/pull/18) (no-sync gradient accumulation). fp32 local-shard gradients need [#20](https://github.com/meta-pytorch/flex_shard/pull/20) (per-parameter `grad_dtype`), and [#21](https://github.com/meta-pytorch/flex_shard/pull/21) keeps their casts as cheap as before; without #20, local-shard gradients stay bf16.

### Flags

| Flag | Effect |
| --- | --- |
| `--use-flex-shard` | Shard parameters over the data-parallel group with FlexShard. Default `reshard_after_forward=True` (ZeRO-3). |
| `--flex-shard-no-reshard-after-forward` | Keep gathered parameters from forward until backward (ZeRO-2). |
| `--flex-shard-no-sync` | With gradient accumulation, reduce-scatter only in the last microbatch's backward. Earlier microbatches accumulate full gradients (fp32 with `--accumulate-allreduce-grads-in-fp32`, the bf16 default), at the memory cost of one full gradient copy. |
| `--flex-shard-no-reshard-after-backward` | With `--flex-shard-no-sync`, keep gathered parameters between microbatches, so only the first microbatch all-gathers (without reshard-after-forward). |

`validate_args` rejects combining `--use-flex-shard` with any of:
- TP, PP or EP > 1
- `--use-distributed-optimizer`, `--overlap-param-gather`
- gradient-accumulation fusion (it writes `main_grad`, which only exists under Megatron DDP)
- tied embeddings
- fp16
- optimizers other than Adam / SGD
- `--use-torch-fsdp2` or `--use-megatron-fsdp`

### Example

```bash
torchrun --nproc-per-node 8 pretrain_gpt.py \
  --num-layers 24 --hidden-size 2048 --ffn-hidden-size 5632 --num-attention-heads 16 \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 1 --global-batch-size 8 --train-iters 60 \
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 5 --clip-grad 1.0 \
  --bf16 --swiglu --normalization RMSNorm --position-embedding-type rope \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --transformer-impl transformer_engine --no-gradient-accumulation-fusion \
  --mock-data --tokenizer-type NullTokenizer --vocab-size 32000 \
  --log-throughput --timing-log-level 1 --eval-iters 0 \
  --use-flex-shard --flex-shard-no-reshard-after-forward
```

For the Megatron baseline, replace the last line with `--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather`.

Without TransformerEngine, use `--transformer-impl local` together with:
- `--no-rope-fusion`
- `--no-bias-swiglu-fusion`
- `--no-masked-softmax-fusion`
- `--no-bias-dropout-fusion`
- `--no-persist-layer-norm`

## Design

`FlexShardDataParallel` (`megatron/core/distributed/flex_shard_data_parallel.py`) subclasses `_BaseDataParallel`, like the torch FSDP2 wrapper.

- **Buckets**, in forward order: the embedding, one bucket per `TransformerLayer`, any remaining parameter-owning modules (e.g. `final_layernorm`), then `output_layer`. Each bucket is one all-gather before use and one reduce-scatter after backward. Every parameter is `Shard(0)` (`per_param_placements`) over the `dp_cp` group. Buckets are split by dtype, because FlexShard requires one dtype per bucket.
- **Reshard-after-forward** follows `FlexShardDataParallelConfig.reshard_after_forward`. The last bucket never reshards, because its backward runs immediately (like the FSDP2 root).
- **Parameter attributes.** FlexShard replaces each parameter with a local-shard tensor. The wrapper saves Megatron's per-parameter attributes (`tensor_model_parallel`, `allreduce`, ...) before `flex_shard()` and restores them afterwards.
- **Optimizer.** After wrapping, `module.parameters()` yields local shards, so Megatron's existing `Float16OptimizerWithFloat16Params` (Adam, fp32 main params) updates only this rank's shard. With `--accumulate-allreduce-grads-in-fp32` (the bf16 default), each bf16 parameter gets `grad_dtype=torch.float32` before `flex_shard()`, so FlexShard stores fp32 local-shard gradients, as the distributed optimizer keeps them, and the optimizer uses them as its main gradients without a copy. Each gradient element lives on exactly one data-parallel rank, so grad stats (norm, zero count) are reduced over WORLD (`megatron/core/optimizer/__init__.py`).
- **Grad sync.** FlexShard reduce-scatters during backward and waits at the end of backward, so `finish_grad_sync` is a no-op. `scale_gradients` scales the local shards.
- **No-sync.** With `--flex-shard-no-sync`, `train()` puts `FlexShardDataParallel.no_sync` in `no_sync_func`. It turns FlexShard's `set_requires_gradient_sync` off on entry and back on at exit, as Megatron DDP's `no_sync()` does with `is_last_microbatch`; FlexShard, like FSDP2, has no context manager of its own. Backwards of all but the last microbatch keep full gradients on FlexShard's persistent unsharded parameters, and autograd accumulates into them. The last microbatch's backward reduce-scatters them, including buckets it did not use. FlexShard always reshards after that syncing backward, so `--flex-shard-no-reshard-after-backward` cannot leave stale parameters after the optimizer step.
- **Selection.** `get_megatron_ddp_config` returns a `FlexShardDataParallelConfig` when `--use-flex-shard` is set. Both `get_model()` and the ModelBuilder path that `pretrain_gpt.py` uses (`megatron/training/models/dist_utils.py:_ddp_wrap`) pick the wrapper from that config type.
- **Process groups** come from `pg_collection.dp_cp`, with a fallback to `parallel_state` for callers that don't pass groups.

### Limitations

- **Checkpoint save/load** is not wired up.
- **`torch.compile`:** FlexShard falls back to synchronous unshard under compile, so it is not used here.

## Benchmark: Megatron DDP + DistributedOptimizer vs FlexShard (`reshard_after_forward=False`)

The goal is an apples-to-apples comparison between Megatron's data-parallel baseline and FlexShard configured to behave like it. AdamW comes first (Phase A), then Muon (Phase B). The [Plan](#plan) lists the steps; the results so far come first.

### Method

The baseline is **M1**: Megatron DDP + DistributedOptimizer (ZeRO-1).
- It keeps full bf16 params during forward and backward.
- It shards the optimizer state.
- Per step, it reduce-scatters the gradients once and all-gathers the params once.

**F2** is FlexShard with `reshard_after_forward=False` (ZeRO-2).
- It also keeps full bf16 params during forward and backward.
- With gradient accumulation, `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` makes it move the same bytes per step as M1.

A gap between M1 and F2 therefore measures the implementation (scheduling and bucketing), not the sharding strategy.

| | M1: DDP + DistOpt | F2: FlexShard, `reshard_after_forward=False` |
| --- | --- | --- |
| Flags | `--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather` | `--use-flex-shard --flex-shard-no-reshard-after-forward`, plus `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` with gradient accumulation |
| Comm per step | RS grads + AG params (after the optimizer, overlapped with the next forward) | AG per bucket in the first forward + RS per bucket in the last backward |
| Optimizer | `DistributedOptimizer` (Adam, fp32 main params, 1/DP) | `Float16OptimizerWithFloat16Params` (Adam, fp32 main params) on local 1/DP shards |
| Grads | persistent full fp32 grad buffer, reduce-scattered to fp32 shards | fp32 local-shard grads, plus full fp32 grads between microbatches only |
| Buckets | ~40M-param contiguous buckets | one per `TransformerLayer` + embedding / final norm / output |

Both arms match in:
- the model, data and seed;
- bf16 params, fp32 grads and fp32 main params;
- Adam with decoupled weight decay and global-norm clipping;
- gradient accumulation through `no_sync_func`.

FlexShard's fp32 shard grads need flex_shard #20 (see [Requirements](#requirements)).

Reference arms (not part of the goal):
- **M0:** plain DDP, to show what the distributed optimizer adds.
- **F1** (FlexShard ZeRO-3, `reshard_after_forward=True`) and **M2** (Megatron-FSDP `optim_grads_params`). These trade communication for memory that M1 doesn't save, so they are compared only with each other (see the [Roadmap](#roadmap)).

Setup:
- 8x H100 96 GB, DP = 8, TP = PP = 1.
- Mock data, bf16 params, MBS 1, seq 2048, lr 3e-4.
- Timing is the median over iterations ≥ 20.

Shapes:
- **S:** 24 layers, hidden 2048, ffn 5632, 16 heads (1.5B).
- **L:** 32 layers, hidden 4096, ffn 14336, GQA 32/8 (8B). It runs with `--recompute-granularity selective` on every arm, because M1 runs out of memory without it.

### Results (local spec, no TransformerEngine)

These were measured with FlexShard before flex_shard #16 and with bf16 FlexShard shard grads. Phase A re-measures them.

| Shape | M1 ms/it | F2 ms/it | Δ | M1 / F2 max allocated |
| --- | --- | --- | --- | --- |
| S | 157.7 | 156.6 | −1% | 21.5 / 16.5 GB |
| L | 594.9 | 538.1 | −10% | 59.4 / 33.6 GB |

- **Profile (S, rank 0, one step):** compute is the same in both arms (~117 ms). F2 exposes 45.5 ms of NCCL (AG 25.3 / RS 17.6) vs 13.0 ms for M1. F2's stalls are ~1 ms gaps before `split_with_sizes_copy_out`: each layer waits for its own all-gather, and one-bucket-ahead prefetch does not hide it.
- **GA = 4, before no-sync (not like-for-like):** S 495.8 vs 545.3 ms (+10%); L 1824.4 vs 2006.2 ms (+10%). The extra per-microbatch cost at S (+17 ms) matches F2's exposed reduce-scatter. See [Gradient accumulation with no-sync](#gradient-accumulation-with-no-sync-te-s-dp--4) for the fix.
- **ZeRO-3 pair (GA = 1):** F1 vs M2 is 179.9 vs 164.1 ms (S) and 649.6 vs 533.0 ms (L), with similar memory.
- **Correctness:**
  - Iteration-1 loss and grad norm are bit-identical across M0/M1/M2/F1/F2 (local spec) and across M1/F1/F2 (TE spec).
  - A 4-layer model tracks over 20 iterations (loss 2.357658 vs 2.357765).
  - S and L diverge after ~iteration 5 in every arm, M0 vs M1 included, because lr 3e-4 is unstable for them.

### Gradient accumulation with no-sync (TE, S, DP = 4)

Setup:
- TransformerEngine spec, shape S, 4x H100 with no other jobs, MBS 1.
- GBS 8 means 2 microbatches per step, and GBS 32 means 8.
- Two GPU sets ran the two GBS series at the same time. Each cell is the mean of two interleaved repetitions of the median ms/it over iterations ≥ 20.

Arms:
- `+ns` adds `--flex-shard-no-sync`.
- `+ns+keep` also adds `--flex-shard-no-reshard-after-backward`.

| Arm | GBS 8 ms/it | GBS 32 ms/it | max allocated |
| --- | --- | --- | --- |
| M1 (DDP + DistOpt) | 192.6 | 640.7 (612.9 / 668.5) | 12.9 GB |
| F2 | 216.2 | 795.9 | 9.0 GB |
| F2 +ns | 206.7 | 710.4 | 13.5 GB |
| F2 +ns+keep | **193.2** | **606.3** | 13.5 GB |
| F1 | 239.2 | 875.4 | 6.6 GB |
| F1 +ns | 228.4 | 792.1 | 11.0 GB |
| F1 +ns+keep | 215.4 | 703.8 | 11.0 GB |

- **F2 +ns+keep matches M1.** At 2 microbatches it is +0.3% vs M1, and at 8 microbatches −5% vs M1's mean (−1% vs its faster repetition). Like M1, it does one reduce-scatter and one all-gather per bucket per step.
- **No-sync alone** cuts F2 by 4% (GBS 8) and 11% (GBS 32). Keeping parameters cuts another 7% and 15%, because each later microbatch skips the forward all-gather.
- **F1 (ZeRO-3)** still re-gathers every bucket in each backward. With +ns+keep it is 12% behind M1 at 2 microbatches and 10% at 8.
- **Memory:** no-sync adds about 4.5 GB at the peak, for the full fp32 gradients it keeps between microbatches. Keeping parameters adds nothing at the peak. F2 +ns+keep uses 0.7 GB more than M1.
- **Correctness:** iteration-5 loss and grad norm agree across all arms within run-to-run noise. For example, at GBS 8: M1 10.09253 / 61.682, F2 +ns 10.09292 / 61.558, F2 +ns+keep 10.09299 / 61.576.

### Plan

#### Phase A: AdamW, M1 vs F2

1. **Matched configuration.** Use the [Method](#method) table. This needs flex_shard #20 and #21 for fp32 shard grads. The Megatron side is already in: `--accumulate-allreduce-grads-in-fp32` (the bf16 default) gives bf16 params fp32 shard grads.
2. **Correctness gate.**
   - Run shape S at DP = 4, then L at DP = 8, with at least 2 microbatches for about 500 steps.
   - Use a stable lr, e.g. 1e-4 with a 50-step warmup; 3e-4 diverges after about 5 iterations in every arm.
   - Run M1 twice for the noise floor, since TransformerEngine kernels are not bit-deterministic.
   - F2 passes if its loss and grad norm stay within the M1-vs-M1 spread.
3. **Performance.**
   - Cover S and L at 1, 2 and 8 microbatches with the TransformerEngine spec, and also seq 4096.
   - For each cell, take the median ms/it over iterations ≥ 20. Report the min and median of at least 3 interleaved repetitions, plus peak allocated memory.
   - Run on a quiet node, or report GPU kernel time when the node is busy.
   - Profile one step per arm for exposed NCCL time and gaps, including M1 at L, where M1 was slower than both sharded arms.
4. **Close gaps and record.**
   - Make one targeted fix per F2 shortfall, e.g. prefetch depth, or bucket size against M1's `--ddp-bucket-size` 20/40/80M with nccl-tests at both sizes.
   - Record the final numbers here.

#### Phase B: Muon, M3 vs F3

F3 keeps `reshard_after_forward=False` on the FlexShard side.

| | M3: Megatron layer-wise Muon | F3: FlexShard ZeRO-2 + DistMuon |
| --- | --- | --- |
| Wiring | `--optimizer muon --use-distributed-optimizer --muon-scalar-optimizer adam` → `LayerWiseDistributedOptimizer` | owned buckets from flex_shard's `materialize_dist_muon_buckets` + `build_local_dist_muon(DistMuon)` (torchtitan `torchtitan/distributed/flex_shard/dist_muon.py`) |
| Ownership | whole matrices, LPT bin-packing per bucket | whole matrices / block groups, `assign_matrices` |
| Comm per step | RS to owners + AG params | AG in forward + RS to owners |
| Optimizer comm | none | none (storage == compute layout) |
| Uneven shards | padded to the largest owner | padded to the largest owner |
| Non-matrix params | Adam | Adam on `Shard(0)` buckets |

- M3 reduce-scatters to owners by default (`use_layer_wise_param_layout=True`); the class docstring's all-reduce flow is the legacy path. It needs `emerging_optimizers` `v0.3.0`.
- M1 and F2 serve as references, so that (M3 − M1) vs (F3 − F2) isolates the cost of switching from Adam to Muon in each stack.

1. **Wire F3 in Megatron.**
   - Pick Muon params with Megatron's `is_managed_by_layer_wise_optimizer` (qkv, proj, fc1 and fc2 weights), so both stacks use the same set.
   - Add a `--flex-shard-dist-muon` mode: `assign_matrices` turns the Muon params into owned buckets per layer, and non-matrix params go on `Shard(0)` buckets.
   - Optimizer: a `ChainedOptimizer` of DistMuon on fp32 main copies of the owned shards, plus Megatron Adam on the rest. Take the global grad norm over WORLD, and handle ranks that own no matrices.
   - Relax the `--use-flex-shard` optimizer restriction.
   - Since flex_shard #20, DistMuon's local adapter requires real-param grads in the param dtype. `--accumulate-allreduce-grads-in-fp32` gives bf16 params fp32 shard grads, so either let DistMuon accept fp32 grads or keep bf16 grads for Muon params.
2. **Parity gate before timing.**

   | Knob | Megatron | DistMuon |
   | --- | --- | --- |
   | NS coefficients / steps | `--muon-coefficient-type`, `--muon-num-ns-steps` | `ns_coefficients` (3.4445, −4.7750, 2.0315), `ns_steps=5` |
   | NS precision | `--muon-fp32-matmul-prec` | bf16 |
   | Update scale | `--muon-scale-mode spectral` | `adjust_lr_fn` (`original` √max(1, r/c), `match_rms_adamw`, `spectral_unclamped`) |
   | Momentum / weight decay | `--muon-momentum`, `--muon-nesterov`, confirm decoupled WD | 0.95, Nesterov, decoupled |
   | QKV / fc1 split | per-head split by default (`--muon-no-split-qkv`) | whole matrix or `BlockShard`; start with no split on both sides |

   Checks:
   - A single-matrix update agrees within ~2e-2.
   - M3 and F3 loss curves agree over 50 iterations at a stable lr.
   - M1 and M3 curves differ, confirming Muon is active.
3. **Performance.** Same protocol as Phase A, step 3, also reporting (M3 − M1) vs (F3 − F2).
4. **Risks.**
   - Coarse ownership at DP = 8 (mitigate with block groups).
   - DistMuon requires a grad for every configured param.
   - Megatron's interleaved per-group QKV layout.
   - Keep `overlap_param_gather_with_optimizer_step` off.

## Roadmap

These are not part of the benchmark goal:
- **Distributed checkpoint save/load** for FlexShard shards and their optimizer state, which real training runs need. flex_shard `426e2bf` adds DCP metadata for model tensors.
- **ZeRO-3: F1 vs Megatron-FSDP (M2),** for models that don't fit with full params resident. The F1 and M2 numbers above predate flex_shard #16.
- **TP, EP and PP,** which `validate_args` rejects today.
