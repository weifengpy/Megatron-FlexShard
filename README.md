# FlexShard data-parallel backend for Megatron-LM

This fork of NVIDIA/Megatron-LM (based on `16251ac12`) adds [FlexShard](https://github.com/meta-pytorch/flex_shard) as a data-parallel backend behind `--use-flex-shard`, alongside DDP, the distributed optimizer, torch FSDP2 and Megatron-FSDP. With the flag off, Megatron behaves like upstream, so one `pretrain_gpt.py` command compares Megatron DDP with FlexShard by flipping a flag.

## Benchmark: Megatron DDP vs FlexShard

Phases A (AdamW) and B (Muon) compare **Megatron DDP + distributed optimizer** (`--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather`, ZeRO-1) with **FlexShard without reshard-after-forward** (`--use-flex-shard --flex-shard-no-reshard-after-forward`, ZeRO-2, plus `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` with gradient accumulation). Both keep full bf16 parameters through forward and backward and move the same bytes per step, so a gap measures the implementation, not the sharding strategy. They share the model, data and seed, fp32 gradients and main parameters, and Adam with global-norm clipping; Megatron uses ~40M-parameter buckets, FlexShard one per layer. Phases C and D bring in Megatron-FSDP (see [Plan](#plan)).

Setup: 8x H100 96 GB, TP = PP = 1, mock data. The **DeepSeek-V3 model** has DeepSeek-V3's layers at full width, fewer of them and with fewer experts: hidden 7168; multi-head latent attention (128 heads, q-lora rank 1536, kv-lora rank 512, qk head dim 128 plus rope head dim 64, v head dim 128, q/kv layernorms); 8 layers, 1 dense (FFN 18432) and 7 MoE, plus 1 MTP layer; 32 routed experts (FFN 2048; DeepSeek-V3 has 256) with top-8 sigmoid routing, expert bias and top-4 of 8 groups, plus 1 shared expert; untied embeddings, all-to-all dispatcher.

### Phase A results on the DeepSeek-V3 model (DP 8, EP 4)

TransformerEngine spec with gradient accumulation fusion on both sides, `CUDA_DEVICE_MAX_CONNECTIONS=1`, lr 1e-4, and expert parallelism: dense parameters shard over all 8 ranks, and experts split 4 ways by expert parallelism, with expert data parallelism 2, so FlexShard shards each expert over 2 ranks (its dense and expert buckets use its default `BucketedBlockShard` layout). Runs are GPU-bound at seq 4096, micro-batch size 1 and 4 microbatches (GBS 32): a profiled step keeps the GPU busy 98% of the time on some stream. Each run is 10 iterations, timed as the median ms/it over iterations 4–10; a cell is the median of two interleaved runs. Both stacks do the same work per step: Megatron adds `--ddp-average-in-collective --no-check-for-nan-in-loss-and-grad` and FlexShard `--no-check-for-nan-in-loss-and-grad` (see below).

| Case | Megatron DDP + distributed optimizer, ms/it | FlexShard |
| --- | --- | --- |
| GPU-bound, 4 microbatches (GBS 32) | 1659.0 | 1639.4 (−1.2%) |
| Same, Megatron with 250M-parameter buckets | 1647.1 | 1626.5 (−1.3%) |

- **Matched work:** before each reduce-scatter, Megatron's defaults scale the gradients by 1/DP in a separate kernel (FlexShard averages inside the collective, as `--ddp-average-in-collective` does) and check them for NaN, which the flags above turn off; `--no-check-for-nan-in-loss-and-grad` also turns off a loss check both stacks run. Expert gradients still get a separate kernel in Megatron, which scales them by 2/8 (expert data parallelism over DP) before the expert data-parallel average: 11.5 ms of GPU time per step on rank 0.
- **Megatron with bigger buckets:** the first row uses Megatron's default buckets, about 40M parameters (31 dense and 56 expert buckets on this model). The second uses `--ddp-bucket-size 250000000` (10 and 11 buckets), from three interleaved runs per stack in a later session. Each row pairs runs from one session, since step times drift by about 0.8% between sessions. Bigger buckets don't close the gap: compared iteration by iteration, Megatron is slower at all 21 timed iterations. With 500M-parameter buckets (6 and 6), one run took 1662.0 ms/it.
- **Correctness:** iterations 1 and 2 match Megatron's loss and grad norm exactly. Over the 10 iterations, the largest loss difference from Megatron is 8.6e-4, against 1.2e-3 between Megatron's two runs.
- **Memory:** FlexShard peaks at 73.7 GB per rank, against Megatron's 72.0 GB.

### Phase B results on the DeepSeek-V3 model (DP 8, EP 4)

The DeepSeek-V3 model's Phase A setup, matched work included, with `--optimizer muon` on both sides: Megatron DDP + layer-wise distributed Muon vs FlexShard. Both run Megatron's `TensorParallelMuon` (Emerging-Optimizers v0.3.0, Megatron's pin) on whole matrices on their owner ranks, so they compute the same update; Megatron all-gathers parameters after the step, FlexShard in the next forward. torchtitan's DistMuon computes a different update, so FlexShard runs Megatron's. Expert matrices are owned within the expert data-parallel group of 2.

| Case | Megatron DDP + layer-wise distributed Muon, ms/it | FlexShard |
| --- | --- | --- |
| GPU-bound, 4 microbatches (GBS 32) | 1990.2 | 1926.0 (−3.2%) |

- **Buckets:** FlexShard merges consecutive MoE layers' non-expert parameters (latent attention, shared expert, router, norms) and MTP's matrices into shared Muon buckets across layers, as it does for dense layers. That takes its owned Muon buckets from 18 to 10. Unmerged, each MoE layer's bucket held a few large matrices whose owner rows were mostly padding (about 4x the real size), and FlexShard ran out of memory.
- **Correctness:** iterations 1 and 2 match Megatron's loss and grad norm exactly. Over the 10 iterations, FlexShard's two runs differ from Megatron's four (from two sessions) by up to 8.8e-4 in loss, against up to 1.0e-3 between Megatron's own runs.
- **Memory:** FlexShard peaks at 70.9 GB per rank, against Megatron's 76.3 GB.

## Usage

### Requirements

- PyTorch with CUDA and NCCL. Tested with a PyTorch 2.15 dev build on CUDA 13; flex_shard declares `torch>=2.14,<2.15`, but its tests pass on 2.15.
- flex_shard at or after the `pyproject.toml` pin (`main` at `582a126`, [#47](https://github.com/meta-pytorch/flex_shard/pull/47)). Installing it with dependencies pulls its `torch<2.15` pin. With a checkout, `pip install --no-deps -e <flex_shard>` or `PYTHONPATH=<flex_shard>/src` also work; then install `torchao` separately.
- Gradient accumulation fusion (Megatron's default) needs APEX's `fused_weight_gradient_mlp_cuda` for Megatron's own linear layers, including the GPT output layer under the TransformerEngine spec, with or without FlexShard.
- On Hopper, TransformerEngine's grouped-tensor GEMM (`--moe-use-grouped-tensor`, which single grouped MoE weights need) needs cuBLAS 13.4+ (13.6+ with blockwise FP8), both when TransformerEngine is built and at run time. Otherwise per-expert weights silently fall back to split GEMMs, and single grouped weights raise an error. The grouped-tensor runs here used TransformerEngine 2.21.0.dev0 built against the `nvidia-cublas` 13.8 wheel and loaded through `LD_PRELOAD`, since PyTorch's and TransformerEngine's RPATHs take precedence over `LD_LIBRARY_PATH`.

### Flags

| Flag | Effect |
| --- | --- |
| `--use-flex-shard` | Shard parameters over the data-parallel group with FlexShard. Reshards after forward by default (ZeRO-3). |
| `--flex-shard-no-reshard-after-forward` | Keep gathered parameters from forward until backward (ZeRO-2). |
| `--flex-shard-no-sync` | With gradient accumulation, reduce-scatter only in the last microbatch's backward. Earlier microbatches accumulate full gradients, at the memory cost of one full gradient copy. |
| `--flex-shard-no-reshard-after-backward` | With `--flex-shard-no-sync`, keep gathered parameters between microbatches, so without reshard-after-forward only the first microbatch all-gathers. |
| `--flex-shard-placement {bucketed-block,shard0}` | Layout of each bucket without one of its own: `bucketed-block` (default, flex_shard's `BucketedBlockShard`) or `shard0` (`Shard(0)`), see [Design](#design). Muon and single grouped MoE weights always use `shard0`. |

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
- **Bucket layout.** By default, each bucket except FP8 ones is one parameter-major buffer cut into equal contiguous per-rank ranges at row boundaries (flex_shard's `BucketedBlockShard`), like the distributed optimizer's buffers. After the first gather, unshards gather straight into the buffer the parameters view, and buckets of fused weight gradients reduce-scatter the buffer their `main_grad` views, so neither copies. With `--flex-shard-placement shard0`, Muon or single grouped MoE weights, buckets cut each parameter by rows (`Shard(0)`) instead, as do buckets holding a weight tied across pipeline stages (below). FP8 and Muon matrix buckets have their own placements (below).
- **Parameters.** FlexShard replaces each parameter with its local shard; the wrapper saves Megatron's per-parameter attributes (`tensor_model_parallel`, `allreduce`, ...) and restores them.
- **Optimizer.** Megatron's `Float16OptimizerWithFloat16Params` updates the local shards. With `--accumulate-allreduce-grads-in-fp32` (the bf16 default), bf16 parameters get `grad_dtype=torch.float32`, so FlexShard's fp32 local-shard gradients serve as main gradients without a copy. Each gradient element lives on one data-parallel rank, so grad stats (norm, zero count) reduce over WORLD.
- **Grad sync.** FlexShard reduce-scatters during backward and waits at its end, so `finish_grad_sync` is a no-op and `scale_gradients` scales the local shards. Buckets average over their group, as Megatron DDP scales by 1/DP, or sum with `--calculate-per-token-loss`, where `finalize_model_grads` divides by the global token count.
- **No-sync.** `--flex-shard-no-sync` puts `FlexShardDataParallel.no_sync` in `no_sync_func`, which toggles FlexShard's `set_requires_gradient_sync` as Megatron DDP's `no_sync()` does. Earlier microbatches accumulate full gradients on FlexShard's persistent unsharded parameters. The last microbatch's backward reduce-scatters every bucket and always reshards, so kept parameters can't go stale across the optimizer step.
- **Gradient accumulation fusion.** Fused linear layers add weight gradients into `param.main_grad` and give autograd none. For their buckets, a flex_shard pre-backward hook allocates the gathered parameters' gradients (zeroed, and fp32 with `--accumulate-allreduce-grads-in-fp32`) as views of a buffer that mirrors the gathered buffer the parameters view, and aliases them as `main_grad`, and a post-reduce hook drops the aliases. The fused GEMMs thus add straight into the gradients FlexShard reduce-scatters; with `BucketedBlockShard` and Muon buckets, that buffer is in the gathered bucket's layout, so the reduce-scatter reads it as is. Under full activation recompute, whose original forward runs without gradients, flex_shard runs that pre-backward hook when it unshards the bucket for the recomputed forward.
- **Tied embeddings.** The output layer reuses the embedding's weight at call time, which FlexShard can't see. The wrapper puts the final norm in the embedding bucket, as torchtitan groups `[tok_embeddings, norm, lm_head]`, and asserts that FlexShard hooks it at the model root. The bucket stays gathered from the embedding through the output layer and skips reshard-after-forward.
- **Tensor parallelism.** FlexShard shards each rank's TP slices over its `dp_cp` group, which excludes its TP peers, and Megatron's TP communication (sequence parallelism, `--tp-comm-overlap`) is unchanged. `finalize_model_grads` all-reduces sequence-parallel and `--qk-layernorm` layer-norm gradients on `param.grad`, FlexShard's local-shard gradient.
- **Expert parallelism.** Each MoE layer's expert parameters (`allreduce=False`) get their own bucket on the expert data-parallel group, hooked on the experts module so its all-gather overlaps attention. Their summed gradients are divided by the dense data-parallel size (`gradient_divide_factor`), as Megatron DDP, Megatron-FSDP and torchtitan scale expert gradients.
  - With `--overlap-dispatch-backward-with-experts-wgrad`, TransformerEngine computes expert weight gradients later, in `backward_dw()`. Their buckets use flex_shard's `defer_post_backward` and finish from each weight's `post_wgrad_grad_acc_hook`.
  - With `--overlap-moe-expert-parallel-comm`, the schedule calls the layers' sub-modules directly, so FlexShard's forward hooks never run. The schedule gathers buckets with flex_shard's `unshard()`, FlexShard finalizes backward manually, and each layer's buckets reduce-scatter from the schedule's per-layer post-backward hook (`set_fsdp_reshard_hooks`, as for Megatron-FSDP), the rest in `start_grad_sync`.
- **Pipeline parallelism.** Each model chunk is its own `FlexShardDataParallel`. With `--flex-shard-no-sync` and the default `--align-grad-reduce`, stages after the first reduce-scatter in `start_grad_sync`, outside backward, through flex_shard's `finalize_backward(async_op=True)`, overlapping the pipeline bubble as Megatron DDP does. With tied embeddings, `finalize_model_grads` all-reduces the two stages' copies on their local-shard gradients, so their buckets keep `Shard(0)`, which cuts both copies alike; `BucketedBlockShard`'s cuts depend on the rest of each bucket.
- **Context parallelism** needs nothing new. Dense buckets shard over `dp_cp` and expert buckets over the expert data-parallel group, which both include CP, and gradients scale as Megatron DDP's do. CP needs `--transformer-impl transformer_engine`, whose attention does CP's communication.
- **Blockwise FP8 parameter all-gather.** With `--fp8-param-gather --fp8-recipe blockwise`, FlexShard all-gathers the weights of TransformerEngine's linear layers in FP8, halving their all-gather bytes and gathered memory, while local shards, gradients and the optimizer stay bf16/fp32. Weights with dims that are multiples of 128 use `TEBlockwiseFp8Shard` (`te_fp8.py`), a subclass of flex_shard's `Fp8BucketedBlockShard` that cuts 128-row block rows, quantizes them with TransformerEngine's quantizer and builds a `Float8BlockwiseQTensor` over the gathered buffer. It overrides three private flex_shard methods for TransformerEngine's scale layout, so it must be re-synced when flex_shard changes them. Numerics equal Megatron's blockwise FP8 training without `--fp8-param-gather`.
- **Single grouped MoE weights.** TransformerEngine's `GroupedTensor` parameter (`--moe-single-grouped-weight`, `--moe-single-grouped-bias`, with `NVTE_GROUPED_LINEAR_SINGLE_PARAM=1`) bans shape ops. So before sharding, `te_grouped.py` replaces it with a plain (experts × out, in) parameter over the same buffer, and each forward `TEGroupedLinear._get_weight_tensors` wraps the gathered tensor back into a `GroupedTensor`, both without copies.
- **Muon.** With `--optimizer muon`, buckets holding Muon matrices store each matrix whole on one rank (flex_shard's `BucketedOwned`), with owners balanced by size over all such buckets. Consecutive layers share such a bucket, closed as Megatron's layer-wise distributed Muon closes its buckets, at max(`--ddp-bucket-size`, 0.9 × DP × the bucket's largest matrix), so each rank's padded share stays near the mean; the bucket's patterns name those layers, so flex_shard hooks it on their forwards. The reduce-scatter delivers each whole gradient to its owner, which runs Megatron's `TensorParallelMuon` with no optimizer communication, and the next forward's all-gather carries the update. As with `BucketedBlockShard`, neither collective copies: each rank's local shards are laid out as its row of the gathered bucket, which the all-gather sends as is, the parameters view the gathered rows, and buckets of fused weight gradients reduce-scatter the buffer their `main_grad` views. Other parameters keep row shards and Adam.
- **Checkpoints** use `torch_dist` with Megatron DDP's keys and global shapes, so the two load each other's checkpoints. `checkpoint_adapter.py` runs the module's own `sharded_state_dict()` on meta stand-ins of the full parameters and maps each entry onto the local shard with a `ShardedTensorFactory`, one uneven piece per overlap with flex_shard's layouts. The optimizer reuses it for main parameters, Adam moments and Muon momentum.

### Limitations

- Checkpoints need `--ckpt-format torch_dist`, and don't support FP8 parameter all-gather or single grouped MoE biases yet.
- Under `torch.compile`, FlexShard falls back to synchronous unshard, so compile isn't used here.

## Plan

**Phase A (AdamW)** is done on the DeepSeek-V3 model; see its results above. Next:
- a 200-iteration correctness gate on the DeepSeek-V3 model, for AdamW and Muon;
- fixes for FlexShard's per-bucket CPU work.

Protocol: the median ms/it over iterations ≥ 20, the min and median of at least 3 interleaved repetitions, peak allocated memory, and one profiled step per setup. For correctness, a stable lr (1e-4 with warmup; 3e-4 diverges in every setup) and two Megatron runs as the noise floor, since TransformerEngine kernels aren't bit-deterministic.

**Phase B (Muon)** is done on the DeepSeek-V3 model; see its results above.

**Phase C:** Megatron-FSDP (`--use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params`) vs FlexShard with reshard-after-forward, both ZeRO-3, for models that don't fit with full parameters resident. Measure with Phase A's protocol, then close gaps.

**Phase D:** Megatron-FSDP v2, experimental since June 2026 (Megatron-LM [#5387](https://github.com/NVIDIA/Megatron-LM/pull/5387)), is converging on FlexShard's design: per-parameter placements, including whole-tensor ownership like `BucketedOwned`, explicit rank layouts with uneven collectives ([#7670](https://github.com/NVIDIA/Megatron-LM/pull/7670), [#7671](https://github.com/NVIDIA/Megatron-LM/pull/7671)), and owner-based compute planning ([#6597](https://github.com/NVIDIA/Megatron-LM/pull/6597)). Compare it once it trains end to end in Megatron, against Phase C's setup and, if it runs Muon, Phase B's.

## Composition with Megatron features

These were checked on small dense and MoE test models. Unless noted, FlexShard's iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly. They used `Shard(0)`, the default before `BucketedBlockShard`; `BucketedBlockShard` also matches exactly, through the iteration-2 loss, at DP 8, TP 2 with sequence parallelism, PP 2 with and without virtual chunks and with MTP, CP 2, MTP, full recompute, fusion off, blockwise FP8 parameter all-gather, and at EP 4 with delayed expert weight gradients and with the EP overlap.

- **Tensor parallelism:** TP 2 × DP 2, with and without sequence parallelism, no-sync, `--qk-layernorm` and `--tp-comm-overlap`.
- **Gradient accumulation fusion:** with and without full recompute, and with MoE recompute, including grouped-tensor experts.
- **Tied embeddings:** DP 4 and TP 2 × DP 2 with sequence parallelism, with no-sync, fusion and MTP.
- **Expert parallelism:** EP 1, 2, 4 and 8; TP 2 with sequence parallelism, expert TP 1 and 2, and EP 2 and 4; and EP 4 with no-sync, per-token loss, expert bias, shared experts, a dense first layer, MTP, MoE recompute, the allgather dispatcher, fusion, experts without grouped GEMM, and 64 experts with top-1 routing. Grouped-tensor experts (`--moe-use-grouped-tensor`) match at EP 1, 2, 4 and 8, at TP 2, at PP 2 × EP 2, and at EP 4 with options including delayed expert weight gradients, the EP overlap and blockwise FP8 parameter all-gather; their checkpoints at EP 4 resume within 1.1e-3 of an uninterrupted run, about as far as two MoE runs drift, and load Megatron's with its next loss exactly.
- **Pipeline parallelism:** 8 microbatches at PP 2 × DP 4, PP 4 × DP 2, PP 2 × TP 2 × DP 2 and PP 2 with 2 virtual chunks, with no-sync and syncing every microbatch; at PP 2 × DP 4 also with MTP, per-token loss, full recompute, an uneven layer split, fusion and untied embeddings; and PP 2 × EP 2, with and without TP 2. With later stages reduce-scattering in the pipeline bubble, they still match at PP 2 × DP 2 (with and without virtual chunks, with MTP) and at PP 2 × EP 2. Still to do: a benchmark, with and without the bubble overlap.
- **Context parallelism:** an 8K sequence at CP 2 × DP 4, with and without no-sync. Still to do: CP with TP, PP and EP; CP 4 and 8; the all-gather, all-to-all and hierarchical CP communication types; MTP, recompute, fusion, per-token loss and untied embeddings; longer loss curves; and a long-context benchmark.
- **Blockwise FP8 parameter all-gather,** against Megatron's blockwise FP8 training without `--fp8-param-gather`: at DP 8, with less peak memory than FlexShard's bf16 all-gather. The iteration-1 loss and grad norm and the iteration-2 loss also match with TP 2, PP 2, per-token loss, full recompute and fusion, at EP 2 (with and without the EP overlap), and with grouped-tensor experts at EP 4. Later iterations differ by up to 5.4e-4 on the dense test model, and by 1.8e-3 to 3.2e-3 on the MoE test model, where two Megatron runs differ by 1.2e-3 to 1.3e-3. Other recipes still switch `--fp8-param-gather` off with a warning. Still to do: loss curves against Megatron with `--fp8-param-gather`, and a benchmark.
- **Single grouped MoE weights:** a standalone `GroupedLinear` check is bit-identical to TransformerEngine's own single grouped parameters, with and without fusion. They match at EP 1, 2, 4 and 8, at TP 2 with sequence parallelism, at PP 2 × EP 2, and at EP 4 syncing every microbatch, without fusion, with single grouped biases, MTP, MoE recompute, delayed expert weight gradients and the EP overlap, with the same memory as per-expert weights. Their checkpoints resume within 1.3e-3 and cross-load with Megatron's exactly. Still to do: a benchmark against per-expert weights.
- **Muon:** the iteration-1 loss and grad norm and the iteration-2 loss, the first after a Muon step, match Megatron's layer-wise distributed Muon exactly at DP 8, with reshard-after-forward, full recompute and PP 2; an earlier composition matrix (syncing every microbatch, without fusion, untied embeddings, QKV without the split, MTP, PP 2, TP 2 with sequence parallelism, EP 4) matched through iteration 2. Checkpoints at DP 8 resume at iteration 5 bit-exactly over iterations 6–10, Muon momentum and Adam moments included, as Megatron's own resume does, and Megatron and FlexShard load each other's weights (`--no-load-optim`) with the next loss and grad norm exact.
- **Checkpoints:** at DP 8, FlexShard resumes at iteration 5 within 1.1e-4 of an uninterrupted run over iterations 6–10 (Megatron's own resume: 8.7e-5). Megatron and FlexShard load each other's weights (`--no-load-optim`) with the next loss exact. Still to do: loading at a different DP size, TP or PP, and a mid-run resume on the DeepSeek-V3 model.
- **Evaluation** (`--eval-iters > 0`) is untested; every benchmark ran with evaluation off.
