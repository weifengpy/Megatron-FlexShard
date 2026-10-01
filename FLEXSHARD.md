# FlexShard data-parallel backend for Megatron-LM

This fork adds [FlexShard](https://github.com/meta-pytorch/flex_shard) as a Megatron data-parallel backend behind `--use-flex-shard`, alongside DDP, the distributed optimizer, torch FSDP2 and Megatron-FSDP. With the flag off, Megatron behaves exactly like upstream, so one `pretrain_gpt.py` command compares Megatron's DDP / DistributedOptimizer with FlexShard by flipping a flag.

Base: upstream NVIDIA/Megatron-LM `16251ac12` plus one commit, "Add FlexShard data-parallel backend (--use-flex-shard)".

## Usage

### Requirements

- PyTorch with CUDA and NCCL. Tested with a PyTorch 2.15 dev build on CUDA 13 (flex_shard declares `torch>=2.14,<2.15`, but its test suite passes on 2.15).
- `flex_shard` importable (`pip install --no-deps -e <flex_shard>` or `PYTHONPATH=<flex_shard>/src`). It also needs `torchao`, which is not declared: `flex_shard.custom_placements` imports the fp8 placement unconditionally.
- TransformerEngine layers (`--transformer-impl transformer_engine`) need flex_shard with these draft PRs applied, in order:
  - [#13](https://github.com/meta-pytorch/flex_shard/pull/13): keep `self.weight` readable in backward (TE's `RMSNorm` re-reads it).
  - [#14](https://github.com/meta-pytorch/flex_shard/pull/14): expose unsharded params through `module._parameters` during forward (TE's operation fuser passes `op.parameters()` as autograd inputs).
  - [#15](https://github.com/meta-pytorch/flex_shard/pull/15): re-unshard on backward-time reads with reshard-after-forward (ZeRO-3).
  - The local spec (`--transformer-impl local`) works without them.

### Flags

| Flag | Effect |
| --- | --- |
| `--use-flex-shard` | Shard parameters over the data-parallel group with FlexShard. Default `reshard_after_forward=True` (ZeRO-3). |
| `--flex-shard-no-reshard-after-forward` | Keep gathered parameters from forward until backward (ZeRO-2). |

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
- **Optimizer.** After wrapping, `module.parameters()` yields local shards, so Megatron's existing `Float16OptimizerWithFloat16Params` (Adam, fp32 main params) updates only this rank's shard. Each gradient element lives on exactly one data-parallel rank, so grad stats (norm, zero count) are reduced over WORLD (`megatron/core/optimizer/__init__.py`).
- **Grad sync.** FlexShard reduce-scatters during backward and waits at the end of backward, so `finish_grad_sync` is a no-op. `scale_gradients` scales the local shards.
- **Selection.** `get_megatron_ddp_config` returns a `FlexShardDataParallelConfig` when `--use-flex-shard` is set. Both `get_model()` and the ModelBuilder path that `pretrain_gpt.py` uses (`megatron/training/models/dist_utils.py:_ddp_wrap`) pick the wrapper from that config type.
- **Process groups** come from `pg_collection.dp_cp`, with a fallback to `parallel_state` for callers that don't pass groups.

### Limitations

- **No no-sync.** Every microbatch's backward reduce-scatters, and with ZeRO-3 every microbatch also re-gathers. Gradient accumulation (GA > 1) therefore moves k times the bytes of Megatron DDP. See the [Roadmap](#roadmap).
- **One transformer layer per bucket.** FlexShard hooks a bucket on the deepest module owning all its parameters. For a multi-layer bucket that is the `decoder.layers` ModuleList, whose forward never runs ("bucket hook did not run").
- **Checkpoint save/load** is not wired up.
- **`torch.compile`:** FlexShard falls back to synchronous unshard under compile, so it is not used here.

## Benchmark: Megatron DDP + DistributedOptimizer vs FlexShard ZeRO-2

### Method

The like-for-like pair at GA = 1 is **M1** (Megatron DDP + DistributedOptimizer, ZeRO-1) vs **F2** (FlexShard ZeRO-2). Both move the same bytes per step (one reduce-scatter of grads and one all-gather of params) and hold full bf16 params during forward/backward. They differ only in scheduling and bucketing, so a gap measures implementation, not sharding strategy.

| | M1: DDP + DistOpt | F2: FlexShard ZeRO-2 |
| --- | --- | --- |
| Comm per step | RS grads + AG params (after the optimizer, overlapped with the next forward) | AG per bucket in forward + RS per bucket in backward |
| Optimizer | `DistributedOptimizer` (Adam, fp32 main, 1/8) | `Float16OptimizerWithFloat16Params` (Adam, fp32 main) on local 1/8 shards |
| Grads | full bf16 grad buffer per rank | sharded |
| Buckets | ~40M-param contiguous buckets | one per `TransformerLayer` + embedding / final norm / output |

Reference arms:
- M0: plain DDP.
- M2: Megatron-FSDP `optim_grads_params`, compared only with F1.
- F1: FlexShard ZeRO-3, compared only with M2.

Setup:
- 8x H100 96 GB, DP = 8, TP = PP = 1.
- Mock data, bf16 grads, MBS 1, seq 2048, lr 3e-4.
- Timing is the median over iterations ≥ 20.

Shapes:
- **S:** 24 layers, hidden 2048, ffn 5632, 16 heads (1.5B).
- **L:** 32 layers, hidden 4096, ffn 14336, GQA 32/8 (8B). It runs with `--recompute-granularity selective` on every arm, because M1 runs out of memory without it.

### Results (local spec, no TransformerEngine)

| Shape | M1 ms/it | F2 ms/it | Δ | M1 / F2 max allocated |
| --- | --- | --- | --- | --- |
| S | 157.7 | 156.6 | −1% | 21.5 / 16.5 GB |
| L | 594.9 | 538.1 | −10% | 59.4 / 33.6 GB |

- **Profile (S, rank 0, one step):** compute is the same in both arms (~117 ms). F2 exposes 45.5 ms of NCCL (AG 25.3 / RS 17.6) vs 13.0 ms for M1. F2's stalls are ~1 ms gaps before `split_with_sizes_copy_out`: each layer waits for its own all-gather, and one-bucket-ahead prefetch does not hide it.
- **GA = 4 (feature gap, not like-for-like):** S 495.8 vs 545.3 ms (+10%); L 1824.4 vs 2006.2 ms (+10%). The extra per-microbatch cost at S (+17 ms) matches F2's exposed reduce-scatter.
- **ZeRO-3 pair (GA = 1):** F1 vs M2 is 179.9 vs 164.1 ms (S) and 649.6 vs 533.0 ms (L), with similar memory.
- **Correctness:**
  - Iteration-1 loss and grad norm are bit-identical across M0/M1/M2/F1/F2 (local spec) and across M1/F1/F2 (TE spec).
  - A 4-layer model tracks over 20 iterations (loss 2.357658 vs 2.357765).
  - S and L diverge after ~iteration 5 in every arm, M0 vs M1 included, because lr 3e-4 is unstable for them.

TransformerEngine runs work (M1, F1, F2), but no trustworthy TE timings yet: the shared machine had other jobs on some GPUs, which put ranks out of step.

### Remaining benchmark work

1. Profile M1 vs F2 at L (M1 is slower than both sharded arms there).
2. Loss parity at a stable lr, and a parameter-equality check after N steps.
3. M1 `--ddp-bucket-size` 20/40/80M, MBS 2, and nccl-tests at both bucket sizes.
4. TE timings at seq 4096 on a quiet machine.

## Roadmap

### No-sync gradient accumulation (FlexShard)

Closing the GA > 1 gap needs no-sync (one reduce-scatter per step), plus keeping full params between microbatches (one all-gather per step, like FSDP2's `set_reshard_after_backward(False)`).

- **FSDP2:** `set_requires_gradient_sync(False)` skips the reduce-scatter in `post_backward`. Autograd then accumulates into the persistent unsharded parameter's `.grad`.
- **FlexShard** needs an explicit per-bucket accumulator, because its unsharded params are outputs of the `_BucketUnshard` autograd function, not leaves. Full grads arrive in `_BucketUnshard.backward` and are reduce-scattered immediately. Planned design:
  - **API:** `FlexShardModule.set_requires_gradient_sync(bool, buckets=None)` and a `no_sync()` context manager. Optional `BucketSpec(accumulate_dtype=...)` for fp32 accumulation.
  - **No-sync microbatches:** grads `add_` into the accumulator (the first microbatch keeps the grad tensor without copying).
  - **Syncing microbatch:** adds into the accumulator, then reduce-scatters through the existing path.
  - **Unused params:** an end-of-backward callback reduces buckets that hold accumulated grads but got none in the last microbatch.
  - **Phase 2:** accumulate in the flat reduce-scatter layout, so the last microbatch skips the copy-in (like Megatron's contiguous grad buffer).
  - **Memory:** full unsharded grads between microbatches (S: 3 GB bf16, L: 16 GB), which is Megatron DDP's cost. Hence opt-in, per bucket.
  - **Megatron wiring:** `FlexShardDataParallel.no_sync()`, add the wrapper to `no_sync_func` setup in `training.py`, and a `--flex-shard-no-sync` flag.
  - **Expected gain:** F2 at GA = 4 drops from ~545 to ~495 ms (M1 parity).
- **Alternative:** an FSDP2-style persistent unsharded parameter, with `resize_(0)` storage on reshard, would provide no-sync through autograd and cover all backward-time parameter reads. It's a larger redesign of FlexShard's core.

### Muon: Megatron layer-wise Muon (M3) vs FlexShard ZeRO-2 + DistMuon (F3)

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

Wiring F3:
1. Pick Muon params with Megatron's `is_managed_by_layer_wise_optimizer` (qkv, proj, fc1, fc2 weights), so both stacks use the same set.
2. Add a `--flex-shard-dist-muon` mode: `assign_matrices` → owned buckets per layer, with non-matrix params on `Shard(0)` buckets.
3. Optimizer: a `ChainedOptimizer` of DistMuon on fp32 main copies of the owned shards, plus Megatron Adam on the rest. Global grad norm over WORLD. Handle ranks that own no matrices.
4. Relax the `--use-flex-shard` optimizer restriction.

Parity gate before timing:

| Knob | Megatron | DistMuon |
| --- | --- | --- |
| NS coefficients / steps | `--muon-coefficient-type`, `--muon-num-ns-steps` | `ns_coefficients` (3.4445, −4.7750, 2.0315), `ns_steps=5` |
| NS precision | `--muon-fp32-matmul-prec` | bf16 |
| Update scale | `--muon-scale-mode spectral` | `adjust_lr_fn` (`original` √max(1, r/c), `match_rms_adamw`, `spectral_unclamped`) |
| Momentum / weight decay | `--muon-momentum`, `--muon-nesterov`, confirm decoupled WD | 0.95, Nesterov, decoupled |
| QKV / fc1 split | per-head split by default (`--muon-no-split-qkv`) | whole matrix or `BlockShard`; start with no split on both sides |

Checks:
- A single-matrix update agrees within ~2e-2.
- M3 vs F3 loss curves agree over 50 iterations at a stable lr.
- M1 vs M3 curves differ, confirming Muon is active.

Risks:
- Coarse ownership at DP = 8 (mitigate with block groups).
- DistMuon requires a grad for every configured param.
- Megatron's interleaved per-group QKV layout.
- Keep `overlap_param_gather_with_optimizer_step` off.
