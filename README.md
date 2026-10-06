# FlexShard data-parallel backend for Megatron-LM

This fork of NVIDIA/Megatron-LM (based on `16251ac12`) adds [FlexShard](https://github.com/meta-pytorch/flex_shard) as a data-parallel backend behind `--use-flex-shard`, alongside DDP, the distributed optimizer, torch FSDP2 and Megatron-FSDP. With the flag off, Megatron behaves like upstream, so one `pretrain_gpt.py` command compares Megatron DDP with FlexShard by flipping a flag.

## Usage

### Requirements

- PyTorch with CUDA and NCCL. Tested with a PyTorch 2.15 dev build on CUDA 13; flex_shard declares `torch>=2.14,<2.15`, but its tests pass on 2.15.
- flex_shard at or after the `pyproject.toml` pin (`main` at `09dd2dc`, [#40](https://github.com/meta-pytorch/flex_shard/pull/40)). Installing it with dependencies pulls its `torch<2.15` pin. With a checkout, `pip install --no-deps -e <flex_shard>` or `PYTHONPATH=<flex_shard>/src` also work; then install `torchao` separately.
- Gradient accumulation fusion (Megatron's default) needs APEX's `fused_weight_gradient_mlp_cuda` for Megatron's own linear layers, including the GPT output layer under the TransformerEngine spec, with or without FlexShard.
- On Hopper, TransformerEngine's grouped-tensor GEMM (`--moe-use-grouped-tensor`, which single grouped MoE weights need) needs cuBLAS 13.4+ (13.6+ with blockwise FP8), both when TransformerEngine is built and at run time. Otherwise per-expert weights silently fall back to split GEMMs, and single grouped weights raise an error. The grouped-tensor runs here used TransformerEngine 2.21.0.dev0 built against the `nvidia-cublas` 13.8 wheel and loaded through `LD_PRELOAD`, since PyTorch's and TransformerEngine's RPATHs take precedence over `LD_LIBRARY_PATH`.

### Flags

| Flag | Effect |
| --- | --- |
| `--use-flex-shard` | Shard parameters over the data-parallel group with FlexShard. Reshards after forward by default (ZeRO-3). |
| `--flex-shard-no-reshard-after-forward` | Keep gathered parameters from forward until backward (ZeRO-2). |
| `--flex-shard-no-sync` | With gradient accumulation, reduce-scatter only in the last microbatch's backward. Earlier microbatches accumulate full gradients, at the memory cost of one full gradient copy. |
| `--flex-shard-no-reshard-after-backward` | With `--flex-shard-no-sync`, keep gathered parameters between microbatches, so without reshard-after-forward only the first microbatch all-gathers. |
| `--flex-shard-no-bucketed-block-shard` | Cut each parameter by rows (`Shard(0)`) instead of the default `BucketedBlockShard` (see [Design](#design)). Muon and single grouped MoE weights never use `BucketedBlockShard`. |

`validate_args` rejects `--use-flex-shard` with:
- `--use-distributed-optimizer`, `--overlap-param-gather`, `--use-torch-fsdp2`, `--use-megatron-fsdp` or fp16;
- optimizers other than Adam, SGD and Muon, and Muon with FP8 parameter all-gather or single grouped MoE weights;
- `--save` or `--load` with a `--ckpt-format` other than `torch_dist`, with FP8 parameter all-gather, or with single grouped MoE biases;
- `--overlap-moe-expert-parallel-comm` without `--flex-shard-no-reshard-after-forward --flex-shard-no-sync --flex-shard-no-reshard-after-backward`;
- `--delay-wgrad-compute` without gradient accumulation fusion, and `--overlap-dispatch-backward-with-experts-wgrad` with `--flex-shard-no-sync` but without fusion;
- single grouped MoE weights or biases (`--moe-single-grouped-weight`, `--moe-single-grouped-bias`) with the TransformerEngine op fuser, with FP8 or FP4, or with delayed weight gradients but without fusion.

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

For the Megatron baseline, replace the last line with `--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather`. Without TransformerEngine, use `--transformer-impl local --no-rope-fusion --no-bias-swiglu-fusion --no-masked-softmax-fusion --no-bias-dropout-fusion --no-persist-layer-norm`.

## Design

`FlexShardDataParallel` (`megatron/core/distributed/flex_shard/flex_shard_data_parallel.py`) subclasses `_BaseDataParallel`, like the torch FSDP2 wrapper. `get_megatron_ddp_config` returns a `FlexShardDataParallelConfig` under `--use-flex-shard`, and both `get_model()` and the ModelBuilder path that `pretrain_gpt.py` uses (`_ddp_wrap`) pick the wrapper by that type.

- **Buckets**, in forward order: the embedding, one per `TransformerLayer`, any other parameter-owning modules (e.g. `final_layernorm`), then `output_layer`, split by dtype. Each all-gathers before use and reduce-scatters after backward, over `pg_collection.dp_cp`. The last bucket never reshards after forward, like the FSDP2 root.
- **Bucket layout.** By default, each bucket except FP8 ones is one parameter-major buffer cut into equal contiguous per-rank ranges at row boundaries (flex_shard's `BucketedBlockShard`), like the distributed optimizer's buffers. After the first gather, unshards gather straight into the buffer the parameters view, and buckets of fused weight gradients reduce-scatter the buffer their `main_grad` views, so neither copies. With `--flex-shard-no-bucketed-block-shard`, Muon or single grouped MoE weights, buckets cut each parameter by rows (`Shard(0)`) instead, as do buckets holding a weight tied across pipeline stages (below). FP8 and Muon matrix buckets have their own placements (below).
- **Parameters.** FlexShard replaces each parameter with its local shard; the wrapper saves Megatron's per-parameter attributes (`tensor_model_parallel`, `allreduce`, ...) and restores them.
- **Optimizer.** Megatron's `Float16OptimizerWithFloat16Params` updates the local shards. With `--accumulate-allreduce-grads-in-fp32` (the bf16 default), bf16 parameters get `grad_dtype=torch.float32`, so FlexShard's fp32 local-shard gradients serve as main gradients without a copy. Each gradient element lives on one data-parallel rank, so grad stats (norm, zero count) reduce over WORLD.
- **Grad sync.** FlexShard reduce-scatters during backward and waits at its end, so `finish_grad_sync` is a no-op and `scale_gradients` scales the local shards. Buckets average over their group, as Megatron DDP scales by 1/DP, or sum with `--calculate-per-token-loss`, where `finalize_model_grads` divides by the global token count.
- **No-sync.** `--flex-shard-no-sync` puts `FlexShardDataParallel.no_sync` in `no_sync_func`, which toggles FlexShard's `set_requires_gradient_sync` as Megatron DDP's `no_sync()` does. Earlier microbatches accumulate full gradients on FlexShard's persistent unsharded parameters. The last microbatch's backward reduce-scatters every bucket and always reshards, so kept parameters can't go stale across the optimizer step.
- **Gradient accumulation fusion.** Fused linear layers add weight gradients into `param.main_grad` and give autograd none. For their buckets, a flex_shard pre-backward hook allocates each gathered parameter's gradient (zeroed, and fp32 with `--accumulate-allreduce-grads-in-fp32`) and aliases it as `main_grad`, and a post-reduce hook drops the alias. The fused GEMMs thus add straight into the gradient FlexShard reduce-scatters, with no extra buffer or copy. Under activation recompute, each fused layer also sets the alias in a forward pre-hook when its forward runs inside backward.
- **Tied embeddings.** The output layer reuses the embedding's weight at call time, which FlexShard can't see. The wrapper puts the final norm in the embedding bucket, as torchtitan groups `[tok_embeddings, norm, lm_head]`, and asserts that FlexShard hooks it at the model root. The bucket stays gathered from the embedding through the output layer and skips reshard-after-forward.
- **Tensor parallelism.** FlexShard shards each rank's TP slices over its `dp_cp` group, which excludes its TP peers, and Megatron's TP communication (sequence parallelism, `--tp-comm-overlap`) is unchanged. `finalize_model_grads` all-reduces sequence-parallel and `--qk-layernorm` layer-norm gradients on `param.grad`, FlexShard's local-shard gradient.
- **Expert parallelism.** Each MoE layer's expert parameters (`allreduce=False`) get their own bucket on the expert data-parallel group, hooked on the experts module so its all-gather overlaps attention. Their summed gradients are divided by the dense data-parallel size (`gradient_divide_factor`), as Megatron DDP, Megatron-FSDP and torchtitan scale expert gradients.
  - With `--overlap-dispatch-backward-with-experts-wgrad`, TransformerEngine computes expert weight gradients later, in `backward_dw()`. Their buckets use flex_shard's `defer_post_backward` and finish from each weight's `post_wgrad_grad_acc_hook`.
  - With `--overlap-moe-expert-parallel-comm`, the schedule calls the layers' sub-modules directly, so FlexShard's forward hooks never run. The schedule gathers buckets with flex_shard's `unshard()`, FlexShard finalizes backward manually, and each layer's buckets reduce-scatter from the schedule's per-layer post-backward hook (`set_fsdp_reshard_hooks`, as for Megatron-FSDP), the rest in `start_grad_sync`.
- **Pipeline parallelism.** Each model chunk is its own `FlexShardDataParallel`. With `--flex-shard-no-sync` and the default `--align-grad-reduce`, stages after the first reduce-scatter in `start_grad_sync`, outside backward, through flex_shard's `finalize_backward(async_op=True)`, overlapping the pipeline bubble as Megatron DDP does. With tied embeddings, `finalize_model_grads` all-reduces the two stages' copies on their local-shard gradients, so their buckets keep `Shard(0)`, which cuts both copies alike; `BucketedBlockShard`'s cuts depend on the rest of each bucket.
- **Context parallelism** needs nothing new. Dense buckets shard over `dp_cp` and expert buckets over the expert data-parallel group, which both include CP, and gradients scale as Megatron DDP's do. CP needs `--transformer-impl transformer_engine`, whose attention does CP's communication.
- **Blockwise FP8 parameter all-gather.** With `--fp8-param-gather --fp8-recipe blockwise`, FlexShard all-gathers the weights of TransformerEngine's linear layers in FP8, halving their all-gather bytes and gathered memory, while local shards, gradients and the optimizer stay bf16/fp32. Weights with dims that are multiples of 128 use `TEBlockwiseFp8Shard` (`te_fp8.py`), a subclass of flex_shard's `Fp8BucketedBlockShard` that cuts 128-row block rows, quantizes them with TransformerEngine's quantizer and builds a `Float8BlockwiseQTensor` over the gathered buffer. It overrides three private flex_shard methods for TransformerEngine's scale layout, so it must be re-synced when flex_shard changes them. Numerics equal Megatron's blockwise FP8 training without `--fp8-param-gather`.
- **Single grouped MoE weights.** TransformerEngine's `GroupedTensor` parameter (`--moe-single-grouped-weight`, `--moe-single-grouped-bias`, with `NVTE_GROUPED_LINEAR_SINGLE_PARAM=1`) bans shape ops. So before sharding, `te_grouped.py` replaces it with a plain (experts × out, in) parameter over the same buffer, and each forward `TEGroupedLinear._get_weight_tensors` wraps the gathered tensor back into a `GroupedTensor`, both without copies.
- **Muon.** With `--optimizer muon`, buckets holding Muon matrices store each matrix whole on one rank (flex_shard's `BucketedOwned`), with owners balanced by size over all such buckets. The reduce-scatter delivers each whole gradient to its owner, which runs Megatron's `TensorParallelMuon` with no optimizer communication, and the next forward's all-gather carries the update. Other parameters keep row shards and Adam.
- **Checkpoints** use `torch_dist` with Megatron DDP's keys and global shapes, so the two load each other's checkpoints. `checkpoint_adapter.py` runs the module's own `sharded_state_dict()` on meta stand-ins of the full parameters and maps each entry onto the local shard with a `ShardedTensorFactory`, one uneven piece per overlap with flex_shard's layouts. The optimizer reuses it for main parameters and Adam moments.

### Limitations

- Checkpoints need `--ckpt-format torch_dist`, and don't support FP8 parameter all-gather or single grouped MoE biases yet.
- Under `torch.compile`, FlexShard falls back to synchronous unshard, so compile isn't used here.
- With `--overlap-moe-expert-parallel-comm`, `BucketedBlockShard` copies fused weight gradients into its reduce-scatter buffer, since the schedule's `unshard()` allocates the `main_grad` aliases before flex_shard's gradient buckets exist.

## Benchmark: Megatron DDP vs FlexShard

Phases A (AdamW) and B (Muon) compare **Megatron DDP + distributed optimizer** (`--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather`, ZeRO-1) with **FlexShard without reshard-after-forward** (`--use-flex-shard --flex-shard-no-reshard-after-forward`, ZeRO-2, plus `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` with gradient accumulation). Both keep full bf16 parameters through forward and backward and move the same bytes per step, so a gap measures the implementation, not the sharding strategy. They share the model, data and seed, fp32 gradients and main parameters, and Adam with global-norm clipping; Megatron uses ~40M-parameter buckets, FlexShard one per layer. Phases C and D bring in Megatron-FSDP (see [Plan](#plan)).

Setup: 8x H100 96 GB, TP = PP = 1, mock data. Models:
- **117M model:** 4 layers, hidden 1024.
- **1.4B model:** 24 layers, hidden 2048, ffn 5632, 16 heads.
- **7.2B model:** Llama-3-8B layers (32 layers, hidden 4096, ffn 14336, GQA 32/8) with a 32K vocabulary, run with `--recompute-granularity selective`, without which Megatron with the distributed optimizer runs out of memory.
- **Small MoE model:** 4 layers, hidden 1024, 8 experts, top-2, expert FFN 2048, grouped GEMM, all-to-all dispatcher, tied embeddings.
- **10B MoE model:** 16 of Qwen3-30B-A3B's 48 layers.

### Phase A results (1.4B model, DP 8)

TransformerEngine spec with gradient accumulation fusion on both sides, untied embeddings, `CUDA_DEVICE_MAX_CONNECTIONS=1`, lr 1e-4. Each run is 15 iterations, timed as the median ms/it over iterations 7–15, and a cell is the median of two runs (GPU-bound: micro-batch size 4, seq 4096) or three (CPU-bound: micro-batch size 1, seq 2048). `BucketedBlockShard` uses flex_shard #38, #42 and #39.

| Case | Megatron DDP + distributed optimizer, ms/it | FlexShard `Shard(0)` | FlexShard `BucketedBlockShard` |
| --- | --- | --- | --- |
| GPU-bound, 1 microbatch (GBS 32) | 360.3 | 362.4 (+0.6%) | 356.2 (−1.1%) |
| GPU-bound, 4 microbatches (GBS 128) | 1345.2 | 1353.2 (+0.6%) | 1346.0 (+0.1%) |
| CPU-bound, 1 microbatch (GBS 8) | 115.2 | 117.8 (+2.3%) | 111.9 (−2.9%) |
| CPU-bound, 4 microbatches (GBS 32) | 331.7 | 332.1 (+0.1%) | 324.1 (−2.3%) |

- **`BucketedBlockShard` matches or beats Megatron in every case,** and `Shard(0)` is within 0.6% GPU-bound. CPU-bound runs vary by about 2% within a session, so CPU-bound differences under about 3% are noise.
- **Part of `BucketedBlockShard`'s lead is work Megatron's defaults do and FlexShard skips.** Before each reduce-scatter, Megatron scales the gradients by 1/DP in a separate kernel (4.8 ms of GPU time per step; FlexShard averages inside the collective, as `--ddp-average-in-collective` does) and checks them for NaN (2.9 ms and 52 host syncs per step). Without that, Megatron would take an estimated 352.6 and 1337.5 ms/it GPU-bound, putting `BucketedBlockShard` about 1.0% and 0.6% behind (from one profiled step).
- **Copies,** GPU time per profiled step on rank 0: `Shard(0)` spends 7.8–8.0 ms, about two thirds of it copying gradients into the reduce-scatter buffer. `BucketedBlockShard` spends 0.2–0.3 ms, on the embedding and final-norm buckets, which have no fused weight gradients. Before flex_shard #38, #42 and #39 (17.9 → 9.9 → 7.5 → 0.3 ms), it ran 3.0% behind Megatron GPU-bound at 1 microbatch.
- **CPU:** when CPU-bound, both stacks' forwards are bound by kernel launches. FlexShard's forward hooks cost 10.9 ms of CPU per step with `BucketedBlockShard` and 13.1 ms with `Shard(0)`, against Megatron's 5.5 ms (rank 0, 1 microbatch; timing wrappers inflate these), while Megatron's NaN-check host syncs stall its autograd thread.
- **Memory:** FlexShard uses about 5 GB less at 1 microbatch (33.9 vs 38.8 GB GPU-bound, 8.5 vs 13.4 GB CPU-bound): it frees full gradients after the reduce-scatter, while Megatron keeps a persistent gradient buffer. At 4 microbatches, no-sync keeps them between microbatches, and FlexShard uses 0.1–0.2 GB more.
- **With reshard-after-forward** (the `--use-flex-shard` default; GPU-bound at 1 microbatch, two runs each), `BucketedBlockShard` takes 361.6 ms/it against `Shard(0)`'s 371.3 (−2.6%), at 31.4 vs 31.5 GB.
- **Correctness:** iteration-1 loss matches Megatron's exactly in every run. Over 200 iterations at micro-batch size 1 and 4 microbatches, the largest loss difference from Megatron is 7.7e-3 for `Shard(0)` and 4.7e-3 for `BucketedBlockShard`, against 8.6e-3 between two Megatron runs. With `BucketedBlockShard`, checkpoints on the 4-layer model resume within 1e-4 of an uninterrupted run, as Megatron's own resume does, and cross-load with Megatron's exactly.

### Gradient accumulation (1.4B model, DP 4)

TransformerEngine spec without gradient accumulation fusion, 4x H100, micro-batch size 1, 2 and 8 microbatches (GBS 8 and 32). Each cell is the mean of two repetitions of the median ms/it over iterations ≥ 20.

| Setup | GBS 8 ms/it | GBS 32 ms/it | Max allocated |
| --- | --- | --- | --- |
| Megatron DDP + distributed optimizer | 192.6 | 640.7 | 12.9 GB |
| FlexShard without reshard-after-forward | 216.2 | 795.9 | 9.0 GB |
| … with no-sync | 206.7 | 710.4 | 13.5 GB |
| … with no-sync and params kept | **193.2** | **606.3** | 13.5 GB |
| FlexShard with reshard-after-forward | 239.2 | 875.4 | 6.6 GB |
| … with no-sync | 228.4 | 792.1 | 11.0 GB |
| … with no-sync and params kept | 215.4 | 703.8 | 11.0 GB |

With no-sync (`--flex-shard-no-sync`) and params kept (`--flex-shard-no-reshard-after-backward`), FlexShard without reshard-after-forward matches Megatron: +0.3% at 2 microbatches, and −1% to −5% at 8, where Megatron's two repetitions differ by 9%. No-sync costs about 4.5 GB at the peak for the full fp32 gradients. FlexShard with reshard-after-forward stays 10–12% behind, since it re-gathers every bucket in each backward.

## Plan

**Phase A (AdamW).** The 1.4B model results above cover the matched configuration, a 200-iteration correctness gate and performance. Next:
- the matched comparison: `--ddp-average-in-collective --no-check-for-nan-in-loss-and-grad` for Megatron and `--no-check-for-nan-in-loss-and-grad` for FlexShard (the flag also turns off the loss check, which both stacks run);
- the 7.2B model, with a 500-step correctness gate;
- fixes for FlexShard's per-bucket CPU work, and bucket sizes against Megatron's `--ddp-bucket-size`.

Protocol: the median ms/it over iterations ≥ 20, the min and median of at least 3 interleaved repetitions, peak allocated memory, and one profiled step per setup. For correctness, a stable lr (1e-4 with warmup; 3e-4 diverges in every setup) and two Megatron runs as the noise floor, since TransformerEngine kernels aren't bit-deterministic.

**Phase B (Muon):** Megatron DDP + layer-wise distributed Muon (`--optimizer muon --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather`) vs FlexShard without reshard-after-forward with `--optimizer muon`. Both run Megatron's `TensorParallelMuon` (Emerging-Optimizers v0.3.0, Megatron's pin) on whole matrices on their owner ranks, so they compute the same update and move the same bytes; Megatron all-gathers parameters after the step, FlexShard in the next forward, and FlexShard has no full fp32 gradient buffer. torchtitan's DistMuon computes a different update, so FlexShard runs Megatron's.
- Status, on the 117M model at DP 8: iteration-1 loss and grad norm match exactly, and iterations 2–10 differ by up to 1.4e-4 (1.5e-4 with reshard-after-forward), against 9.5e-5 between two Megatron runs. A composition matrix (syncing every microbatch, without fusion, untied embeddings, QKV without the split, MTP, PP 2, TP 2 with sequence parallelism, the small MoE model at EP 4) matches exactly through iteration 2, the first after a Muon step. FlexShard peaks at 1858–1872 MB against Megatron's 1385–1391 MB, from padding: a per-layer bucket has four matrices for eight ranks, so each collective moves about 3.6× the bucket's bytes.
- Next: Muon buckets spanning several layers, which needs a flex_shard option to hook a bucket on a list of modules; the benchmark, with each stack's Muon-minus-AdamW time; Muon checkpoints; and loss curves on the 1.4B model.

**Phase C:** Megatron-FSDP (`--use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params`) vs FlexShard with reshard-after-forward, both ZeRO-3, for models that don't fit with full parameters resident. Measure with Phase A's protocol, then close gaps.

**Phase D:** Megatron-FSDP v2, experimental since June 2026 (Megatron-LM [#5387](https://github.com/NVIDIA/Megatron-LM/pull/5387)), is converging on FlexShard's design: per-parameter placements, including whole-tensor ownership like `BucketedOwned`, explicit rank layouts with uneven collectives ([#7670](https://github.com/NVIDIA/Megatron-LM/pull/7670), [#7671](https://github.com/NVIDIA/Megatron-LM/pull/7671)), and owner-based compute planning ([#6597](https://github.com/NVIDIA/Megatron-LM/pull/6597)). Compare it once it trains end to end in Megatron, against Phase C's setup and, if it runs Muon, Phase B's.

## Composition with Megatron features

Unless noted, iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly. "Within Megatron's spread" means FlexShard's loss curve differs from Megatron's about as much as two Megatron runs differ. These results used `Shard(0)`, the default before `BucketedBlockShard`. `BucketedBlockShard` also matches exactly, through the iteration-2 loss, at DP 8 with and without reshard-after-forward, TP 2 with sequence parallelism, PP 2 with and without virtual chunks and with MTP, CP 2, MTP, full recompute, fusion off, blockwise FP8 parameter all-gather, and on the small MoE model at EP 4 with delayed expert weight gradients and with the EP overlap.

- **Tensor parallelism:** the 117M model at TP 2 × DP 2, with and without sequence parallelism, no-sync, `--qk-layernorm` and `--tp-comm-overlap`. The 1.4B model over 500 iterations at TP 2 × DP 2 and TP 2 × DP 4 with sequence parallelism is within Megatron's spread; at TP 4 × DP 2, where Megatron's runs are bit-identical, its 50-iteration moving average stays within 0.02 of Megatron's. On the 7.2B model at TP 2 × DP 4 (flex_shard #21, fusion off), FlexShard is 14% faster at 1 microbatch (222 vs 259 ms/it, 16.8 vs 30.4 GB) and 3–4.5% faster at 2 and 8 with no-sync, and doesn't need `CUDA_DEVICE_MAX_CONNECTIONS=1`.
- **Gradient accumulation fusion:** the 117M model, with and without full recompute, and the small MoE model with MoE recompute, including grouped-tensor experts. Still to do: the 7.2B model with fusion.
- **Tied embeddings:** the 117M model at DP 4 and at TP 2 × DP 2 with sequence parallelism, with no-sync, fusion and MTP. The 1.4B model over 500 iterations is within Megatron's spread.
- **Expert parallelism:** the small MoE model at EP 1, 2, 4 and 8; at TP 2 with sequence parallelism, expert TP 1 and 2, and EP 2 and 4; and at EP 4 with reshard-after-forward, no-sync, per-token loss, expert bias, shared experts, a dense first layer, MTP, MoE recompute, the allgather dispatcher, fusion, experts without grouped GEMM, and 64 experts with top-1 routing. Grouped-tensor experts (`--moe-use-grouped-tensor`) match at EP 1, 2, 4 and 8, at TP 2, at PP 2 × EP 2, and at EP 4 with options including delayed expert weight gradients, the EP overlap and blockwise FP8 parameter all-gather; their checkpoints at EP 4 resume within 1.1e-3 of an uninterrupted run, about as far as two MoE runs drift, and load Megatron's with its next loss exactly. The 10B MoE model over 500 iterations at EP 4 × expert data-parallel 2, with no-sync over 4 microbatches, is within Megatron's spread. Still to do: a benchmark on the 10B MoE model.
- **Pipeline parallelism:** the 117M model with 8 microbatches at PP 2 × DP 4, PP 4 × DP 2, PP 2 × TP 2 × DP 2 and PP 2 with 2 virtual chunks, with no-sync, with reshard-after-forward and syncing every microbatch; at PP 2 × DP 4 also with MTP, per-token loss, full recompute, an uneven layer split, fusion and untied embeddings; and the small MoE model at PP 2 × EP 2, with and without TP 2. With later stages reduce-scattering in the pipeline bubble, they still match at PP 2 × DP 2 (with and without virtual chunks, with MTP) and for the small MoE model at PP 2 × EP 2. The 1.4B model over 500 iterations at PP 2 × DP 4 and PP 4 × DP 2 is within Megatron's spread. Still to do: a benchmark, with and without the bubble overlap.
- **Context parallelism:** the 117M model at an 8K sequence, CP 2 × DP 4, with and without reshard-after-forward and with no-sync. Still to do: CP with TP, PP and EP; CP 4 and 8; the all-gather, all-to-all and hierarchical CP communication types; MTP, recompute, fusion, per-token loss and untied embeddings; loss curves on the 1.4B model; and a long-context benchmark.
- **Blockwise FP8 parameter all-gather,** against Megatron's blockwise FP8 training without `--fp8-param-gather`: the 117M model at DP 8, with 94 MB less peak memory than FlexShard's bf16 all-gather (1224 vs 1317 MB). The iteration-1 loss and grad norm and the iteration-2 loss also match with TP 2, PP 2, per-token loss, full recompute and fusion, and on the small MoE model at EP 2 (with and without the EP overlap) and with grouped-tensor experts at EP 4. Later iterations differ by up to 5.4e-4 on the 117M model (4.9e-4 at plain DP 8), and by 1.8e-3 to 3.2e-3 on the MoE model, where two Megatron runs differ by 1.2e-3 to 1.3e-3. Other recipes still switch `--fp8-param-gather` off with a warning. Still to do: loss curves against Megatron with `--fp8-param-gather`, and a benchmark.
- **Single grouped MoE weights:** a standalone `GroupedLinear` check is bit-identical to TransformerEngine's own single grouped parameters, with and without fusion. The small MoE model matches at EP 1, 2, 4 and 8, at TP 2 with sequence parallelism, at PP 2 × EP 2, and at EP 4 with reshard-after-forward, syncing every microbatch, without fusion, with single grouped biases, MTP, MoE recompute, delayed expert weight gradients and the EP overlap. Its memory matches per-expert weights (1655.0 vs 1656.0 MB), and its checkpoints resume within 1.3e-3 and cross-load with Megatron's exactly. Still to do: loss curves on the 10B MoE model, and a benchmark against per-expert weights.
- **Checkpoints:** the 117M model at DP 8 resumes at iteration 5 within 1.1e-4 of an uninterrupted run over iterations 6–10, with and without reshard-after-forward (Megatron's own resume: 8.7e-5). Megatron and FlexShard load each other's weights (`--no-load-optim`) with the next loss exact. Still to do: loading at a different DP size, TP or PP, and a mid-run resume on the 1.4B model.
- **Evaluation** (`--eval-iters > 0`) is untested; every benchmark ran with evaluation off.
