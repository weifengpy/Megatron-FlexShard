# FlexShard data-parallel backend for Megatron-LM

This fork adds [FlexShard](https://github.com/meta-pytorch/flex_shard) as a Megatron data-parallel backend behind `--use-flex-shard`, alongside DDP, the distributed optimizer, torch FSDP2 and Megatron-FSDP. With the flag off, Megatron behaves exactly like upstream, so one `pretrain_gpt.py` command compares Megatron's DDP / DistributedOptimizer with FlexShard by flipping a flag.

Base: upstream NVIDIA/Megatron-LM `16251ac12` plus one commit, "Add FlexShard data-parallel backend (--use-flex-shard)".

## Usage

### Requirements

- PyTorch with CUDA and NCCL. Tested with a PyTorch 2.15 dev build on CUDA 13 (flex_shard declares `torch>=2.14,<2.15`, but its test suite passes on 2.15).
- `flex_shard`, which `pyproject.toml` declares as a dependency (flex_shard `main` at `c316223`, [#34](https://github.com/meta-pytorch/flex_shard/pull/34)), so the FlexShard modules import it unconditionally. With a flex_shard checkout, `pip install --no-deps -e <flex_shard>` or `PYTHONPATH=<flex_shard>/src` also work. It also needs `torchao`, which flex_shard declares since [#30](https://github.com/meta-pytorch/flex_shard/pull/30); `--no-deps` and `PYTHONPATH` skip it, so install it separately. Installing with dependencies pulls flex_shard's `torch<2.15` pin.
- TransformerEngine layers (`--transformer-impl transformer_engine`) need flex_shard `main` at or after [#16](https://github.com/meta-pytorch/flex_shard/pull/16) (persistent unsharded parameters, which superseded drafts #13–#15). `--flex-shard-no-sync` and `--flex-shard-no-reshard-after-backward` need `main` at or after [#18](https://github.com/meta-pytorch/flex_shard/pull/18) (no-sync gradient accumulation). fp32 local-shard gradients need `main` at or after [#23](https://github.com/meta-pytorch/flex_shard/pull/23) (per-parameter `grad_dtype`, D121468586); without it, local-shard gradients stay bf16. [#20](https://github.com/meta-pytorch/flex_shard/pull/20) (casts fused into the reduce-scatter copy-in) and [#21](https://github.com/meta-pytorch/flex_shard/pull/21) (deferred upcasts) keep their casts as cheap as before #23.
- Gradient accumulation fusion (Megatron's default; `--no-gradient-accumulation-fusion` turns it off) needs [#25](https://github.com/meta-pytorch/flex_shard/pull/25) (`BucketSpec` pre-backward and post-reduce hooks). Megatron's own linear layers, including the GPT output layer under the TransformerEngine spec, also need APEX's `fused_weight_gradient_mlp_cuda` extension for fusion, with or without FlexShard.
- Expert parallelism needs [#27](https://github.com/meta-pytorch/flex_shard/pull/27) (`BucketSpec.gradient_divide_factor`).
- Delayed expert weight gradients (`--overlap-dispatch-backward-with-experts-wgrad`) need [#33](https://github.com/meta-pytorch/flex_shard/pull/33) (`BucketSpec.defer_post_backward`).
- The EP all-to-all overlap (`--overlap-moe-expert-parallel-comm`) needs [#34](https://github.com/meta-pytorch/flex_shard/pull/34) (`FlexShardModule.unshard`).
- Pipeline parallelism with `--flex-shard-no-sync` needs [#32](https://github.com/meta-pytorch/flex_shard/pull/32) (`finalize_backward`) for the default `--align-grad-reduce`.
- TransformerEngine's grouped-tensor GEMM (`--moe-use-grouped-tensor`, which single grouped MoE weights need) runs on Hopper only with a TransformerEngine built against cuBLAS 13.4+ and with cuBLAS 13.4+ at run time, 13.6+ with blockwise FP8. Otherwise TransformerEngine silently runs per-expert weights on its split GEMMs, and single grouped weights raise an error. This node's CUDA 13.0 and 13.1 ship cuBLAS 13.1 and 13.2. So the grouped-tensor runs below used a TransformerEngine 2.21.0.dev0 rebuilt against the `nvidia-cublas` 13.8 wheel, with that cuBLAS loaded through `LD_PRELOAD`, since PyTorch's and TransformerEngine's RPATHs take precedence over `LD_LIBRARY_PATH`.

### Flags

| Flag | Effect |
| --- | --- |
| `--use-flex-shard` | Shard parameters over the data-parallel group with FlexShard. Default `reshard_after_forward=True` (ZeRO-3). |
| `--flex-shard-no-reshard-after-forward` | Keep gathered parameters from forward until backward (ZeRO-2). |
| `--flex-shard-no-sync` | With gradient accumulation, reduce-scatter only in the last microbatch's backward. Earlier microbatches accumulate full gradients (fp32 with `--accumulate-allreduce-grads-in-fp32`, the bf16 default), at the memory cost of one full gradient copy. |
| `--flex-shard-no-reshard-after-backward` | With `--flex-shard-no-sync`, keep gathered parameters between microbatches, so only the first microbatch all-gathers (without reshard-after-forward). |

`validate_args` rejects combining `--use-flex-shard` with any of:
- `--overlap-moe-expert-parallel-comm` without `--flex-shard-no-reshard-after-forward --flex-shard-no-sync --flex-shard-no-reshard-after-backward`
- `--delay-wgrad-compute` without gradient accumulation fusion, and `--overlap-dispatch-backward-with-experts-wgrad` with `--flex-shard-no-sync` but without it
- TransformerEngine single grouped MoE weights or biases (`--moe-single-grouped-weight`, `--moe-single-grouped-bias`) with the TransformerEngine op fuser, with FP8 or FP4, or with delayed weight gradients but without gradient accumulation fusion
- `--use-distributed-optimizer`, `--overlap-param-gather`
- fp16
- optimizers other than Adam, SGD and Muon, and Muon with FP8 parameter all-gather or single grouped MoE weights
- `--use-torch-fsdp2` or `--use-megatron-fsdp`
- `--save` or `--load` with a `--ckpt-format` other than `torch_dist`, with FP8 parameter all-gather, or with single grouped MoE biases

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

`FlexShardDataParallel` (`megatron/core/distributed/flex_shard/flex_shard_data_parallel.py`) subclasses `_BaseDataParallel`, like the torch FSDP2 wrapper.

- **Buckets**, in forward order: the embedding, one bucket per `TransformerLayer`, any remaining parameter-owning modules (e.g. `final_layernorm`), then `output_layer`. Each bucket is one all-gather before use and one reduce-scatter after backward. Every parameter is `Shard(0)` (`per_param_placements`) over the `dp_cp` group, except expert parameters (see expert parallelism below). Buckets are split by dtype, because FlexShard requires one dtype per bucket.
- **Reshard-after-forward** follows `FlexShardDataParallelConfig.reshard_after_forward`. The last bucket never reshards, because its backward runs immediately (like the FSDP2 root).
- **Parameter attributes.** FlexShard replaces each parameter with a local-shard tensor. The wrapper saves Megatron's per-parameter attributes (`tensor_model_parallel`, `allreduce`, ...) before `flex_shard()` and restores them afterwards.
- **Optimizer.** After wrapping, `module.parameters()` yields local shards, so Megatron's existing `Float16OptimizerWithFloat16Params` (Adam, fp32 main params) updates only this rank's shard. With `--accumulate-allreduce-grads-in-fp32` (the bf16 default), each bf16 parameter gets `grad_dtype=torch.float32` before `flex_shard()`, so FlexShard stores fp32 local-shard gradients, as the distributed optimizer keeps them, and the optimizer uses them as its main gradients without a copy. Each gradient element lives on exactly one data-parallel rank, so grad stats (norm, zero count) are reduced over WORLD (`megatron/core/optimizer/__init__.py`).
- **Grad sync.** FlexShard reduce-scatters during backward and waits at the end of backward, so `finish_grad_sync` is a no-op. `scale_gradients` scales the local shards. Buckets average over their group, as Megatron DDP scales gradients by 1/DP; with `--calculate-per-token-loss` they sum instead, since `finalize_model_grads` divides every gradient by the global token count.
- **No-sync.** With `--flex-shard-no-sync`, `train()` puts `FlexShardDataParallel.no_sync` in `no_sync_func`. It turns FlexShard's `set_requires_gradient_sync` off on entry and back on at exit, as Megatron DDP's `no_sync()` does with `is_last_microbatch`; FlexShard, like FSDP2, has no context manager of its own. Backwards of all but the last microbatch keep full gradients on FlexShard's persistent unsharded parameters, and autograd accumulates into them. The last microbatch's backward reduce-scatters them, including buckets it did not use. FlexShard always reshards after that syncing backward, so `--flex-shard-no-reshard-after-backward` cannot leave stale parameters after the optimizer step.
- **Selection.** `get_megatron_ddp_config` returns a `FlexShardDataParallelConfig` when `--use-flex-shard` is set. Both `get_model()` and the ModelBuilder path that `pretrain_gpt.py` uses (`megatron/training/models/dist_utils.py:_ddp_wrap`) pick the wrapper from that config type.
- **Process groups** come from `pg_collection.dp_cp`, with a fallback to `parallel_state` for callers that don't pass groups.
- **Tensor parallelism.** With TP, each rank's parameters are its TP slices, and FlexShard shards them over the rank's `dp_cp` group, which excludes its TP peers. Megatron's layers keep their own TP communication: column- and row-parallel linears, sequence-parallel all-gathers and reduce-scatters, and TransformerEngine's `--tp-comm-overlap`. The restored `tensor_model_parallel` attributes keep the grad-norm filter (`param_is_not_tensor_parallel_duplicate`) counting TP-replicated parameters, such as layer norms, once. `finalize_model_grads` all-reduces sequence-parallel and `--qk-layernorm` layer-norm gradients over TP on `param.grad`, FlexShard's local-shard gradient, since there is no `main_grad`. On Hopper, Megatron requires `CUDA_DEVICE_MAX_CONNECTIONS=1` with TP, and FlexShard runs with it.
- **Gradient accumulation fusion.** With it, TransformerEngine's and Megatron's linear layers add weight gradients straight into `param.main_grad` and give autograd none. For buckets with such layers, the wrapper passes flex_shard two per-bucket hooks. The pre-backward hook allocates each gathered parameter's gradient, zeroed and fp32 with `--accumulate-allreduce-grads-in-fp32`, and aliases it as `main_grad`; the post-reduce hook drops the alias once the reduce-scatter has taken the gradient. flex_shard itself knows nothing about `main_grad`. The fused GEMMs therefore add into the gradient FlexShard reduce-scatters, across microbatches with no-sync, with no separate buffer or copy. Megatron's own linear layer captures `main_grad` at forward, so its backward now keeps a `main_grad` attached after forward instead of resetting it to forward's `None`. The embedding and final-norm buckets keep ordinary autograd gradients. Activation recompute runs a layer's forward again inside backward, after an original forward without gradients that left the bucket no pre-backward hook, so each fused layer also aliases its gathered parameters' gradients in a forward pre-hook whenever its forward runs inside backward.
- **Tied embeddings.** The output layer reuses the embedding's weight at call time and registers none, so FlexShard cannot see that use from parameter names. When the model ties them, the wrapper puts the final norm's parameters into the embedding bucket, as torchtitan groups `[tok_embeddings, norm, lm_head]` for FSDP2. Their deepest common module is the model root, which also runs the output layer, so FlexShard hooks the bucket there: it gathers before the embedding, stays gathered through the output layer, re-gathers at the start of backward, and reduce-scatters after both uses' gradients have accumulated. The wrapper asserts that anchor, since FlexShard, like FSDP2, cannot detect call-time tying. The bucket skips reshard-after-forward, which at the root would free the weight right before backward re-gathers it. With fusion, the bucket gets the `main_grad` hooks, whose pre-backward hook now runs before the output layer's fused GEMM.
- **Expert parallelism.** Expert parameters (`allreduce=False`, Megatron's marker for the expert topology) exist only on their EP rank and are replicated over the expert data-parallel group (`pg_collection.expt_dp`), so the wrapper gives each MoE layer's experts their own bucket on that group, after the layer's other parameters, as Megatron DDP keeps them in separate buffers. FlexShard hooks it on the experts module, so it gathers after the token dispatch and its all-gather overlaps attention. Each expert's gradient already sums the tokens its EP peers routed to it, so the sum over the expert data-parallel group is divided by the dense data-parallel size (`gradient_divide_factor`), as Megatron DDP, Megatron-FSDP and torchtitan's FSDP2 scale expert gradients. Keying on `allreduce` matches DDP: at EP 1 with ETP = TP, experts are ordinary dense parameters. Megatron builds separate dense and expert optimizers from the restored `allreduce` attribute; both reduce grad stats over WORLD, and `param_is_not_tensor_parallel_duplicate` filters expert duplicates over ETP. The token dispatchers and router buffers (`expert_bias`, token counts) are untouched. With `--overlap-dispatch-backward-with-experts-wgrad`, TransformerEngine leaves the experts' weight gradients to `backward_dw()`, which the token dispatch's backward runs on a side stream after the experts' backward, overlapping the dispatch all-to-all. TransformerEngine marks those weights `skip_backward_post_hook`, so the wrapper gives their buckets flex_shard's `defer_post_backward` and sets each weight's `post_wgrad_grad_acc_hook`, which Megatron calls after `backward_dw()`, to finish the bucket with `finish_deferred_backward`. Megatron DDP waits for those gradients the same way, through TransformerEngine's weight-gradient hooks. Without fusion, `backward_dw()` assigns `param.grad` instead of adding to it, which would overwrite gradients accumulated without sync, so `--flex-shard-no-sync` needs fusion here. With `--overlap-moe-expert-parallel-comm`, Megatron's schedule runs one microbatch's forward alongside another's backward, splits each layer into schedule steps (attention and router, dispatch, experts, combine) whose backwards are separate autograd calls, and calls the layers' sub-modules directly, so FlexShard's forward hooks on `TransformerLayer` and at the model root never run, as Megatron notes for Megatron-FSDP. The schedule therefore gathers every bucket before the step with flex_shard's `unshard()` (setting the fused layers' `main_grad` aliases the pre-backward hook would set), FlexShard runs with manual backward finalization so a backward call finishes nothing, each TransformerLayer's buckets (its dense parameters and its experts) defer their post-backward and finish from the schedule's per-layer post-backward hook (`set_fsdp_reshard_hooks`, which Megatron-FSDP also uses), after the layer's last backward step or its `backward_dw()`. In the step's last backward, that reduce-scatters each layer while the layers before it run their backward, as Megatron DDP overlaps its bucket reduce-scatters; earlier backwards keep accumulating without sync. `start_grad_sync` reduce-scatters the rest after the last backward (per model chunk with virtual PP, where the schedules already call it, and where sync stays off during backward), with `finish_grad_sync` waiting for it. Buckets on modules the schedule still calls (the experts, the embedding) keep their own triggers. This needs FlexShard's apples-to-apples setup (no reshard-after-forward, no-sync, no reshard-after-backward), which keeps full parameters and gradients for the step, as Megatron DDP does. With `--delay-wgrad-compute`, which requires the overlap, TransformerEngine also delays the attention and expert weight gradients to the schedule's `backward_dw()` steps: the expert buckets defer to their weights' `post_wgrad_grad_acc_hook`, which the schedule runs after each `backward_dw()`, and the other buckets reduce-scatter after the step, after every `backward_dw()`. It needs fusion, since without it `backward_dw()` assigns `param.grad`, which would overwrite the gradients accumulated without sync.
- **Pipeline parallelism.** Each model chunk, one per virtual pipeline stage, is its own `FlexShardDataParallel`, with its own buckets, communication streams and learned bucket order. Megatron's schedules run each microbatch's backward as a separate backward call, which FlexShard finishes with its end-of-backward callback. With `--flex-shard-no-sync`, `train()` hands the schedules one `no_sync` per chunk, and they re-enable sync before each chunk's last microbatch backward, which reduce-scatters. With `--align-grad-reduce` (the default), the schedules re-enable sync before the last backward only on the first stage, and the other stages call `start_grad_sync` (as `grad_sync_func`) after a chunk's last microbatch. It reduce-scatters the accumulated gradients outside backward with flex_shard's `finalize_backward(async_op=True)`, so the reduce-scatter overlaps the pipeline bubble, as Megatron DDP's does, and `finish_grad_sync` waits for it. Tied embeddings across stages need nothing new: the last stage's output weight is a separate copy (`shared = True`) that Megatron sets equal to the first stage's embedding at initialization, and `finalize_model_grads` all-reduces the two copies' gradients over the embedding group on the local shards, which both stages shard identically; the grad-norm filter skips the copy. On every stage with the output layer, the model fetches the tied weight before calling the output layer: the embedding's on a stage that holds the embedding (the MTP stage), the output layer's own copy otherwise. The wrapper groups that weight's bucket with the final norm at the model root, as it does without pipeline parallelism, so the weight is gathered before the model fetches it. For an apples-to-apples comparison with Megatron DDP + distributed optimizer, use `--flex-shard-no-reshard-after-forward --flex-shard-no-sync --flex-shard-no-reshard-after-backward`: like DDP, it keeps a stage's full parameters and gradients across the schedule and reduces once per step, and it is torchtitan's default with PP. With reshard-after-forward, FlexShard all-gathers once per microbatch.
- **Context parallelism.** CP splits each sequence over the CP group and leaves the parameters replicated across it, so it needs nothing new in the wrapper. Dense buckets shard over `dp_cp`, which includes the CP ranks, as the distributed optimizer does, and Megatron folds CP into the expert data-parallel group that expert buckets shard over. Each CP rank normalizes its loss by its own token count, and the buckets average over `dp_cp`, as Megatron DDP scales gradients by 1/`dp_cp`; expert buckets divide by the `dp_cp` size, so CP counts there too. With `--calculate-per-token-loss`, the buckets sum and `finalize_model_grads` divides by the token count all-reduced over `dp_cp`. That is torchtitan's normalization: its FSDP2 shards over `(dp_shard, cp)`, sums gradients, and divides the loss by the global token count. TransformerEngine's attention does all of CP's communication, on the CP group inside its own forward and backward, so CP needs `--transformer-impl transformer_engine`. On Hopper, Megatron requires `CUDA_DEVICE_MAX_CONNECTIONS=1` with CP, as with TP, and FlexShard runs with it.
- **Blockwise FP8 parameter all-gather.** With `--fp8-param-gather` and `--fp8-recipe blockwise` (TransformerEngine's `Float8BlockScaling`), FlexShard all-gathers the weights of TransformerEngine's linear layers in FP8, halving their all-gather bytes and gathered memory, while their local shards, gradients and optimizer stay bf16/fp32: `validate_args` keeps `fp8_param` off, so the model has bf16 parameters, unlike Megatron's FP8 primary weights. Weights whose dims are multiples of 128 (`weight`, or a grouped linear's `weight<i>`) use `TEBlockwiseFp8Shard` (`te_fp8.py`), a subclass of flex_shard's `Fp8BucketedBlockShard`, which cuts their bucket into 128-row block rows so a 128 x 128 block never straddles ranks. flex_shard's placement quantizes with torchao and hard-codes its scale layout (one fp32 scale per block) in three private methods; the subclass overrides them with copies that change only the scales per block row, which TransformerEngine pads to a multiple of 4, so it must be re-synced when flex_shard changes them; the rest of the bucket (norms, biases) shares its one all-gather through `MixedBucketPlacement`. Each rank quantizes its block rows with TransformerEngine's weight quantizer for the recipe, so the gathered FP8 data and scales equal TransformerEngine quantizing the full bf16 weight, as its FSDP2 hook gathers them along dim 0. The weight factory builds a `Float8BlockwiseQTensor` over the gathered buffer, which TransformerEngine's layers use as is instead of quantizing the weight. TransformerEngine derives the column-wise copy backward needs from the row-wise data, and the post-reduce hook frees it, since the next all-gather refills the row-wise data with the updated weights. Numerics equal Megatron's blockwise FP8 training without `--fp8-param-gather`; Megatron's flag instead quantizes fp32 main parameters straight to FP8.
- **Single grouped MoE weights.** With `--moe-single-grouped-weight`, TransformerEngine's `GroupedLinear` stores its local experts' weights as one `GroupedTensor` parameter of shape (experts, out, in): a `torch.Tensor` wrapper over one packed buffer that bans slicing and other shape ops, and that its grouped GEMM requires. Megatron DDP keeps the wrapper and repoints the buffer at its own param buffer. FlexShard slices parameters and swaps plain tensors into the module, so before sharding, `te_grouped.py` replaces each `GroupedTensor` with a plain (experts × out, in) parameter over the same buffer, with no copy and with Megatron's parameter attributes, and FlexShard shards its rows like any 2D weight. Every forward, `TEGroupedLinear._get_weight_tensors` wraps FlexShard's gathered tensor back into a `GroupedTensor`, again with no copy, through an autograd function that returns the gradient as a plain tensor. Under gradient accumulation fusion, TransformerEngine writes the weight gradient into `main_grad` and returns none; the view finds the plain parameter's `main_grad`, which FlexShard aliases at pre-backward, after the forward, through TransformerEngine's lazy `get_main_grad` hook. Biases (`--moe-single-grouped-bias`) work the same way, as (experts, out). TransformerEngine ignores both flags unless `NVTE_GROUPED_LINEAR_SINGLE_PARAM=1` is set, and single grouped parameters need its native grouped GEMM, which on Hopper needs cuBLAS 13.4+ both when TransformerEngine is built and at run time.
- **Muon.** With `--optimizer muon`, every bucket holding a Muon matrix (2-D, neither embedding nor output) stores each parameter whole on one rank: flex_shard's `BucketedOwned` (`own_matrices`). Owners balance each rank's total owned size, which its static memory follows, over all such buckets: parameters from every bucket, largest first, go to the rank with the least total so far, with each rank's share of a bucket capped at what balancing the bucket alone gives, so the collectives' padding does not grow. Balancing each bucket on its own put a layer's largest matrix on the same rank in every layer. The reduce-scatter delivers each matrix's whole gradient to its owner, which runs Megatron's own Muon (`TensorParallelMuon` on Emerging-Optimizers) on the complete matrix, including the split of a fused QKV into whole Q, K and V matrices, with no optimizer communication; the next forward's all-gather carries the updated matrices, so nothing is all-gathered after the step. The other ranks hold an empty (0, n) shard of each matrix, which Muon steps as a no-op. Non-matrix parameters keep row shards and Adam. Compared with Megatron's layer-wise distributed Muon, which keeps full parameters and all-gathers them after the step, the collectives move the same bytes, the all-gather moves into forward, and DDP's full fp32 gradient buffer goes away; torchtitan's DistMuon differs in its update rule (Nesterov, a single Newton-Schulz coefficient set, the update scale, bf16 normalization, per-head QKV), so the optimizer here is Megatron's own.
- **Checkpoints.** `--save` and `--load` use Megatron's `torch_dist` format, with Megatron DDP's keys and global shapes, so the two can load each other's checkpoints. Megatron's modules describe each parameter as one piece of a global tensor, from the parameter's shape and the TP/PP/EP ranks, with data-parallel ranks as replicas; under FlexShard a module's parameter is only the local shard. So `FlexShardDataParallel.sharded_state_dict` (`checkpoint_adapter.py`) runs the module's own `sharded_state_dict()` on zero-memory meta stand-ins of the full TP-local parameters, which gives the keys, global shapes, offsets, layer and expert axes and SwiGLU gate/up split Megatron emits for Megatron DDP. Each such entry becomes a `ShardedTensorFactory` over the local shard: its build intersects Megatron's pieces with the chunks flex_shard's `CheckpointableTensor` layouts describe (`get_flex_shard_global_layouts`) and emits one uneven piece per overlap, owned by this data-parallel rank (`axis_fragmentations=None`, which Megatron leaves to DCP to validate), and its merge writes loaded chunks back into a tensor of the shard's shape for `load_state_dict`. Megatron's optimizer reuses the factory for the fp32 main parameters and Adam moments, which have the shard's shape. A single grouped MoE weight comes out of Megatron as one entry per expert, each over the expert's rows of the stand-in (`TEGroupedLinear._split_grouped_checkpoint_tensor` uses `torch.chunk`), in the same per-expert format as per-expert weights; the adapter maps those views back by storage and gathers them into one factory under the module's own `weight` key.

### Limitations

- **Checkpoints** need `--ckpt-format torch_dist`, and don't support FP8 parameter all-gather yet, whose placement doesn't describe where its block rows sit, or single grouped MoE biases, which Megatron splits into per-expert rows with `unbind`.
- **`torch.compile`:** FlexShard falls back to synchronous unshard under compile, so it is not used here.

## Benchmark: Megatron vs FlexShard

The comparison runs in four phases (see the [Plan](#plan)):
- **Phases A and B** compare Megatron DDP with FlexShard configured to behave like it, apples to apples: Phase A with AdamW, Phase B with Muon.
- **Phase C** brings in Megatron-FSDP, against FlexShard with reshard-after-forward.
- **Phase D** brings in the experimental Megatron-FSDP v2.

The method and the Phase A and B results so far come first.

### Method

For Phases A and B, the baseline is **Megatron DDP with the distributed optimizer** (`--use-distributed-optimizer`, ZeRO-1):
- It keeps full bf16 params during forward and backward.
- It shards the optimizer state.
- Per step, it reduce-scatters the gradients once and all-gathers the params once.

The FlexShard side is **FlexShard without reshard-after-forward** (`--flex-shard-no-reshard-after-forward`, ZeRO-2):
- It also keeps full bf16 params during forward and backward.
- With gradient accumulation, `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` makes it move the same bytes per step as the baseline.

A gap between the two therefore measures the implementation (scheduling and bucketing), not the sharding strategy.

| | Megatron DDP + distributed optimizer | FlexShard without reshard-after-forward |
| --- | --- | --- |
| Flags | `--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather` | `--use-flex-shard --flex-shard-no-reshard-after-forward`, plus `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` with gradient accumulation |
| Communication per step | Reduce-scatter of grads + all-gather of params (after the optimizer, overlapped with the next forward) | All-gather per bucket in the first forward + reduce-scatter per bucket in the last backward |
| Optimizer | `DistributedOptimizer` (Adam, fp32 main params, 1/DP) | `Float16OptimizerWithFloat16Params` (Adam, fp32 main params) on local 1/DP shards |
| Grads | Persistent full fp32 grad buffer, reduce-scattered to fp32 shards | fp32 local-shard grads, plus full fp32 grads between microbatches only |
| Buckets | ~40M-param contiguous buckets | One per `TransformerLayer` + embedding / final norm / output |

Both sides match in:
- the model, data and seed;
- bf16 params, fp32 grads and fp32 main params;
- Adam with decoupled weight decay and global-norm clipping;
- gradient accumulation through `no_sync_func`.

FlexShard's fp32 shard grads need flex_shard #23 (see [Requirements](#requirements)).

Reference setups (not part of the goal):
- **Megatron DDP** without the distributed optimizer, to show what the distributed optimizer adds.
- **FlexShard with reshard-after-forward** (ZeRO-3, the `--use-flex-shard` default). It trades communication for memory that the baseline doesn't save; its counterpart, Megatron-FSDP, comes in [Phase C](#phase-c-megatron-fsdp-vs-flexshard-with-reshard-after-forward).

Setup:
- 8x H100 96 GB, DP = 8, TP = PP = 1.
- Mock data, bf16 params, MBS 1, seq 2048, lr 3e-4.
- Timing is the median over iterations ≥ 20.

Models:
- **1.4B model:** 24 layers, hidden 2048, ffn 5632, 16 heads (1.36B parameters).
- **7.2B model:** 32 layers, hidden 4096, ffn 14336, GQA 32/8, i.e. Llama-3-8B layers with a 32K vocabulary (7.24B parameters). It runs with `--recompute-granularity selective` in every setup, because Megatron with the distributed optimizer runs out of memory without it.

### Phase A results (TransformerEngine, 1.4B model, DP 8)

Setup:
- TransformerEngine spec, bf16 params with fp32 grads and main params, Adam.
- Gradient accumulation fusion on both sides, with APEX's `fused_weight_gradient_mlp_cuda`.
- Untied embeddings, no FP8, `CUDA_DEVICE_MAX_CONNECTIONS=1`, lr 1e-4.
- Megatron DDP + distributed optimizer vs FlexShard without reshard-after-forward, with no-sync and params kept (see [Method](#method)).
- One run per cell, 15 iterations. Time is the median ms/it over iterations 7–15, and memory is the max allocated over the run on every rank.
- Iteration-1 loss matches exactly in every cell.

**GPU-bound: micro-batch size 4, seq 4096** (16,384 tokens per micro-batch per rank)

| Microbatches | Megatron, ms/it | FlexShard, ms/it | Change | Tokens/s per GPU, Megatron / FlexShard | Max allocated per rank, Megatron / FlexShard |
| --- | --- | --- | --- | --- | --- |
| 1 (GBS 32) | 364.1 | 362.9 | −0.3% | 45,000 / 45,150 | 38.8 / 33.9 GB |
| 4 (GBS 128) | 1355.8 | 1365.1 | +0.7% | 48,340 / 48,010 | 38.8 / 39.0 GB |

- **FlexShard matches Megatron.** Compute is the same in both, about 120 ms of forward and 220 ms of backward per micro-batch.
- **Little communication is exposed.** At 1 microbatch, Megatron exposes 1.2 ms of all-gather and 4.5 ms of reduce-scatter, against FlexShard's 2.6 and 4.8 ms.
- **At 4 microbatches,** FlexShard waits 9.9 ms at the end of the last backward for its reduce-scatters to drain, which is the +0.7%.
- **FlexShard's collectives still run slower, hidden behind compute.**
  - All-gathers run at about half Megatron's bandwidth: 101 vs 199 GB/s at 1 microbatch.
  - Reduce-scatters run at 57–66%: 134 vs 237 GB/s, and 95 vs 144 GB/s at 4 microbatches.
  - In isolation, this node does the same all-gather in 0.30 ms (about 300 GB/s) and the reduce-scatter in 0.56 ms (about 320 GB/s).
  - The gap would surface where communication is harder to hide: across nodes, at larger DP, or with less compute per step.
- **Memory:** FlexShard uses 13% less without no-sync's full gradients (1 microbatch), and the same with them.

**CPU-bound contrast: micro-batch size 1, seq 2048**

Two sessions on different days, one run per cell each:

| Microbatches | Session | Megatron, ms/it | FlexShard, ms/it | Change |
| --- | --- | --- | --- | --- |
| 1 (GBS 8) | first | 116.3 | 130.5 | +12.2% |
| 1 (GBS 8) | second | 128.7 | 122.8 | −4.6% |
| 4 (GBS 32) | first | 352.2 | 369.5 | +4.9% |
| 4 (GBS 32) | second | 362.9 | 379.7 | +4.6% |

Max allocated per rank is 13.4 GB for Megatron in every cell. For FlexShard it is 8.5 GB at 1 microbatch and 13.6 GB at 4, where no-sync keeps full gradients.

- **At 1 microbatch, single runs don't settle the comparison.** FlexShard was 12% slower in one session and 5% faster in the next.
  - The forward is bound by kernel launches in both stacks. With 2048 tokens per micro-batch, a layer's forward is about 0.8 ms of GPU work but 2.3–2.8 ms of CPU work to launch.
  - Each forward all-gather then waits for the slowest rank. Once every rank has launched it, it takes 0.25 ms, but ranks start it a median of 2.1–2.6 ms apart.
  - In each run, the same rank arrives last for every all-gather, and it is a different rank in each run. So the step runs at the pace of whichever rank's CPU lags in that run.
  - These figures come from profiled steps; the profiler's shape recording adds some CPU time.
- **At 4 microbatches, FlexShard was about 5% slower in both sessions.** In the first session's profile, the gap was the last backward's reduce-scatters: FlexShard's ran at about 78 GB/s against Megatron's 203, and the step ended with a 36.5 ms drain against 12 ms.

CPU time per train step on rank 0, at 1 microbatch, over steps 7–15. These were measured with timing wrappers instead of the profiler, before flex_shard [#37](https://github.com/meta-pytorch/flex_shard/pull/37):

| | Megatron | FlexShard |
| --- | --- | --- |
| Train step, wall | 120.6 ms | 119.4 ms |
| Forward, main-thread CPU | 57.0 ms | 58.6 ms |
| … of which data-parallel hooks | 5.4 ms (forward pre-hooks on 345 modules) | 13.5 ms (27 bucket pre-forward hooks) |
| Backward data-parallel hooks, on the autograd thread | 23.0 ms (147 per-param hooks, 21.5 ms of it launching 26 reduce-scatters) | 15.0 ms (27 post-backward hooks) |

- **The model dominates the forward's CPU time:** about 45–50 ms per step, roughly 2 ms per layer of Megatron's and TransformerEngine's Python, the same in both stacks.
- **FlexShard's forward hooks cost about 8 ms per step more than Megatron's,** 0.5 vs 0.2 ms per layer. Per bucket:
  - Finishing the unshard takes about 0.2 ms. About half is the copy-out, mostly launching the `split_with_sizes_copy` that `Shard(0)`'s rank-major layout needs. The rest is version-counter and storage bookkeeping. Megatron needs no copy-out, since its params are views into the buffer it all-gathers into.
  - Starting the next bucket's all-gather takes about 0.2 ms: copying the local shards into a send buffer, then launching the collective with its stream, event and profiler-annotation setup.
  - The input-gradient trigger, swapping the parameters in and the post-forward hook take the remaining 0.1 ms.
- **In backward, FlexShard's hooks cost less than Megatron's,** and they run on the autograd thread while the GPU is busy.
- **flex_shard #37 trims the copy-out.** It skips rebuilding full-param views when refilling persistent buffers, as FSDP2 builds its unsharded parameters only once.
  - Averaged over three alternating runs of each, FlexShard's pre-forward hooks drop from 14.2 to 12.8 ms of CPU per step, and the copy-out from 2.9 to 2.0 ms. The ranges across runs don't overlap.
  - Step times moved the same way, but by more than the CPU saved and within their run-to-run spread, so they aren't attributed to #37.
- **A flat-buffer layout doesn't help here.** An experiment replaced `Shard(0)` with flex_shard's `BucketedBlockShard`, which lays each bucket out like Megatron's flat buffer: param-major, with contiguous per-rank ranges. It ran at 138.8 and 394.3 ms/it in the second session, with matching iteration-1 loss.
  - Its forward CPU per layer is the same as `Shard(0)`'s, 2.80 vs 2.75 ms. It drops the per-parameter copy-out, but its own send-buffer setup and bucket copy cost about as much. Most of the per-layer cost is FlexShard's bucket bookkeeping, not the copy.
  - Its reduce path costs more, 5.9 vs 3.6 ms of copy-in CPU per step, because it zero-fills and copies the whole bucket twice.
  - Its slower step times also include stragglers: at 4 microbatches, its forward all-gathers each waited about 2.5 ms, while `Shard(0)`'s in the same session ran at full speed, 288 GB/s.

**Correctness gate** (200 iterations at micro-batch size 1, 4 microbatches, lr 1e-4 with 20 warmup iterations):
- FlexShard's largest loss difference from Megatron, 7.7e-3, is smaller than the largest difference between two Megatron runs, 8.6e-3.
- Its final loss, 0.0077, lies between theirs, 0.0083 and 0.0076.

### Results (local spec, no TransformerEngine)

These were measured with FlexShard before flex_shard #16 and with bf16 FlexShard shard grads. The [Phase A results](#phase-a-results-transformerengine-14b-model-dp-8) supersede them.

| Model | Megatron DDP + distributed optimizer, ms/it | FlexShard without reshard-after-forward, ms/it | Change | Max allocated, Megatron / FlexShard |
| --- | --- | --- | --- | --- |
| 1.4B | 157.7 | 156.6 | −1% | 21.5 / 16.5 GB |
| 7.2B | 594.9 | 538.1 | −10% | 59.4 / 33.6 GB |

- **Profile (1.4B model, rank 0, one step):**
  - Compute is the same in both setups (~117 ms).
  - FlexShard exposes 45.5 ms of NCCL time (all-gather 25.3, reduce-scatter 17.6), against 13.0 ms for Megatron.
  - FlexShard's stalls are ~1 ms gaps before `split_with_sizes_copy_out`: each layer waits for its own all-gather, and one-bucket-ahead prefetch does not hide it.
- **4 microbatches, before FlexShard had no-sync (not like-for-like):**
  - Megatron vs FlexShard: 495.8 vs 545.3 ms (+10%) on the 1.4B model, 1824.4 vs 2006.2 ms (+10%) on the 7.2B model.
  - FlexShard's extra cost per microbatch on the 1.4B model (+17 ms) matches its exposed reduce-scatter.
  - See [Gradient accumulation with no-sync](#gradient-accumulation-with-no-sync-transformerengine-14b-model-dp--4) for the fix.
- **Correctness:**
  - Iteration-1 loss and grad norm are bit-identical with the local spec across Megatron DDP, Megatron with the distributed optimizer, and FlexShard with and without reshard-after-forward.
  - With the TransformerEngine spec, they are bit-identical across Megatron with the distributed optimizer and both FlexShard setups.
  - A 4-layer model tracks over 20 iterations (loss 2.357658 vs 2.357765).
  - The 1.4B and 7.2B models diverge after ~iteration 5 in every setup, Megatron DDP vs Megatron with the distributed optimizer included, because lr 3e-4 is unstable for them.

### Gradient accumulation with no-sync (TransformerEngine, 1.4B model, DP = 4)

Setup:
- TransformerEngine spec, 1.4B model, 4x H100 with no other jobs, MBS 1.
- GBS 8 means 2 microbatches per step, and GBS 32 means 8.
- Two GPU sets ran the two GBS series at the same time. Each cell is the mean of two interleaved repetitions of the median ms/it over iterations ≥ 20.
- "No-sync" means `--flex-shard-no-sync`. "Params kept" means `--flex-shard-no-reshard-after-backward` on top of it.

| Setup | GBS 8 ms/it | GBS 32 ms/it | Max allocated |
| --- | --- | --- | --- |
| Megatron DDP + distributed optimizer | 192.6 | 640.7 (612.9 / 668.5) | 12.9 GB |
| FlexShard without reshard-after-forward | 216.2 | 795.9 | 9.0 GB |
| … with no-sync | 206.7 | 710.4 | 13.5 GB |
| … with no-sync and params kept | **193.2** | **606.3** | 13.5 GB |
| FlexShard with reshard-after-forward | 239.2 | 875.4 | 6.6 GB |
| … with no-sync | 228.4 | 792.1 | 11.0 GB |
| … with no-sync and params kept | 215.4 | 703.8 | 11.0 GB |

- **FlexShard without reshard-after-forward, with no-sync and params kept, matches Megatron.**
  - At 2 microbatches it is +0.3% against Megatron.
  - At 8 microbatches it is −5% against Megatron's mean, and −1% against its faster repetition.
  - Like Megatron, it does one reduce-scatter and one all-gather per bucket per step.
- **No-sync alone** speeds FlexShard without reshard-after-forward up by 4% (GBS 8) and 11% (GBS 32). Keeping params speeds it up by another 7% and 15%, because each later microbatch skips the forward all-gather.
- **FlexShard with reshard-after-forward** (ZeRO-3) still re-gathers every bucket in each backward. With no-sync and params kept, it is 12% behind Megatron at 2 microbatches and 10% at 8.
- **Memory:**
  - No-sync adds about 4.5 GB at the peak, for the full fp32 gradients it keeps between microbatches.
  - Keeping params adds nothing at the peak.
  - FlexShard without reshard-after-forward, with no-sync and params kept, uses 0.7 GB more than Megatron.
- **Correctness:** iteration-5 loss and grad norm agree across all setups within run-to-run noise. For example, at GBS 8:
  - Megatron: 10.09253 / 61.682.
  - FlexShard with no-sync: 10.09292 / 61.558.
  - FlexShard with no-sync and params kept: 10.09299 / 61.576.

### Plan

#### Phase A: AdamW, Megatron DDP vs FlexShard

Megatron DDP + distributed optimizer vs FlexShard without reshard-after-forward, both with Adam.

1. **Matched configuration.** Use the [Method](#method) table. This needs flex_shard #23 for fp32 shard grads, and #20 and #21 to keep their casts cheap. The Megatron side is already in: `--accumulate-allreduce-grads-in-fp32` (the bf16 default) gives bf16 params fp32 shard grads.
2. **Correctness gate.**
   - Run the 1.4B model at DP = 4, then the 7.2B model at DP = 8, with at least 2 microbatches for about 500 steps.
   - Use a stable lr, e.g. 1e-4 with a 50-step warmup; 3e-4 diverges after about 5 iterations in every setup.
   - Run the Megatron baseline twice for the noise floor, since TransformerEngine kernels are not bit-deterministic.
   - FlexShard passes if its loss and grad norm stay within the spread between the two Megatron runs.
3. **Performance.**
   - Cover both models at 1, 2 and 8 microbatches with the TransformerEngine spec, and also seq 4096, with gradient accumulation fusion, Megatron's default, on both sides.
   - For each cell, take the median ms/it over iterations ≥ 20. Report the min and median of at least 3 interleaved repetitions, plus peak allocated memory.
   - Run on a quiet node, or report GPU kernel time when the node is busy.
   - Profile one step per setup for exposed NCCL time and gaps, including Megatron on the 7.2B model, where it was slower than both FlexShard setups.
4. **Close gaps and record.**
   - Make one targeted fix per FlexShard shortfall, e.g. prefetch depth, or bucket size against Megatron's `--ddp-bucket-size` 20/40/80M with nccl-tests at both sizes.
   - Record the final numbers here.
- Status: steps 2 and 3 ran on the 1.4B model, with one run per cell and a 200-iteration gate (see [Phase A results](#phase-a-results-transformerengine-14b-model-dp-8)).
   - When GPU-bound, FlexShard matches Megatron within 1%.
   - When CPU-bound, at 1 microbatch, single runs vary between sessions by more than the gap, because stragglers change from run to run. At 4 microbatches, FlexShard was about 5% slower in both sessions.
   - flex_shard #37 trims FlexShard's per-bucket CPU work by about 1.0–1.4 ms per step.
   - Still to do:
     - the 7.2B model;
     - repetitions for the CPU-bound case, with each rank pinned to its GPU's NUMA node and cores to tame the stragglers;
     - step 4's fixes for FlexShard's lower collective bandwidth and the rest of its per-bucket CPU work.

#### Phase B: Muon, Megatron DDP vs FlexShard

Megatron DDP + layer-wise distributed Muon vs FlexShard without reshard-after-forward. Both run Megatron's own Muon (`TensorParallelMuon` on Emerging-Optimizers) on whole matrices, so they compute the same update.

| | Megatron DDP + layer-wise distributed Muon | FlexShard without reshard-after-forward |
| --- | --- | --- |
| Flags | `--optimizer muon --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather` → `LayerWiseDistributedOptimizer` | `--use-flex-shard --flex-shard-no-reshard-after-forward --optimizer muon`, plus `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` with gradient accumulation |
| Ownership | each parameter whole in one rank's chunk of a DDP bucket, bin-packed by Newton-Schulz cost | each parameter of a bucket holding a Muon matrix whole on one rank (`BucketedOwned`), balanced by size over all such buckets |
| Communication per step | Reduce-scatter of grads to the owners + all-gather of params after the step, overlapped with the next forward | All-gather per bucket in the first forward + reduce-scatter of grads to the owners per bucket in the last backward |
| Optimizer step | Newton-Schulz on the owner, with no communication at TP 1 | Same |
| Padding | each rank's chunk padded to the bucket's largest; buckets span many layers | each rank's share padded to the bucket's largest; one bucket per layer |
| Non-matrix params | Adam, through the regular distributed optimizer | Adam, on their local shards |

- Megatron's layer-wise Muon reduce-scatters to the owning ranks by default (`use_layer_wise_param_layout=True`); the class docstring's all-reduce flow is the legacy path. It needs `emerging_optimizers` `v0.3.0`.
- torchtitan's DistMuon computes a different update (Nesterov, a single Newton-Schulz coefficient set, the update scale, bf16 normalization, per-head QKV), so FlexShard runs Megatron's optimizer instead.
- The two AdamW setups from Phase A serve as references. Comparing each stack's Muon-minus-AdamW difference isolates the cost of switching from Adam to Muon in that stack.
- Status: `--use-flex-shard` runs `--optimizer muon` (see [Design](#design)), with Emerging-Optimizers v0.3.0, Megatron's pin.
   - 117M model at DP 8: iteration-1 loss and grad norm match Megatron's layer-wise distributed Muon exactly, and iterations 2–10 stay within its run-to-run spread, without reshard-after-forward (no-sync; max loss difference 1.4e-4, against 9.5e-5 between two Megatron runs) and with it (1.5e-4). Against Megatron DDP running Muon on every rank, the same optimizer code, the difference is 8e-5.
   - Max allocated memory over 10 iterations, on every rank: 1385–1391 MB for Megatron, 1858–1872 MB for FlexShard without reshard-after-forward and 2220–2227 MB with it. Before owners were balanced across buckets, rank 0 owned every layer's fc1 and peaked at 2025 MB without reshard-after-forward. The remaining gap is padding: a per-layer bucket has four matrices for eight ranks, so the largest matrix sets every rank's padded share and each collective moves about 3.6× the bucket's bytes, more at larger DP. Muon buckets spanning several layers come next.
   - Composition matrix, FlexShard without reshard-after-forward against Megatron's layer-wise Muon:
     - Configurations: syncing every microbatch, without fusion, untied embeddings, QKV without the split, MTP, PP 2, TP 2 with sequence parallelism, and the small MoE model at EP 4.
     - In each, the iteration-1 loss and grad norm and the iteration-2 loss, the first after a Muon step, match exactly.
     - Later iterations differ by 7.3e-5 to 1.6e-4 on the 117M model and by 1.6e-3 on the MoE model.
     - These runs predate the cross-bucket owner balance, which moves matrices between ranks but leaves their updates unchanged.
   - Still to do: checkpoints, loss curves on the 1.4B model, and the benchmark.

1. **Padding.**
   - Use Muon buckets that span several layers, so that each rank's share stays close to the mean, as in Megatron's buckets.
   - FlexShard hooks each bucket on its parameters' deepest common module, which for several layers is the whole `TransformerBlock`. This needs a flex_shard option to hook a bucket on a list of modules.
2. **Performance.** Same protocol as Phase A, step 3, also reporting each stack's Muon-minus-AdamW time and every rank's peak memory.
3. **Checkpoints and loss curves:** Muon optimizer state in `torch_dist`, and loss curves on the 1.4B model over 500 iterations.

#### Phase C: Megatron-FSDP vs FlexShard with reshard-after-forward

Megatron-FSDP (`--use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params`, ZeRO-3) vs FlexShard with reshard-after-forward (the `--use-flex-shard` default, ZeRO-3). Both trade communication for memory that the Phase A baseline doesn't save, so they are compared with each other, for models that don't fit with full params resident.

- Earlier numbers, with the local spec and FlexShard before flex_shard #16, at 1 microbatch:
  - 1.4B model: 179.9 ms for FlexShard vs 164.1 ms for Megatron-FSDP.
  - 7.2B model: 649.6 vs 533.0 ms.
  - Memory was similar, and iteration-1 loss and grad norm were bit-identical to the Phase A setups.

1. **Re-measure** both with the current flex_shard, with Phase A's protocol (step 3).
2. **Close gaps and record,** as in Phase A, step 4.

#### Phase D: Megatron-FSDP v2 vs FlexShard

Megatron-FSDP v2 has been experimental since June 2026 (`megatron_fsdp/experimental/`, Megatron-LM [#5387](https://github.com/NVIDIA/Megatron-LM/pull/5387)), and it is converging on FlexShard's design. Its DBuffer has:
- per-parameter placements: `RowAtomic`, `BlockAtomic`, and `TensorAtomic`, which stores a whole tensor on one rank, like FlexShard's `BucketedOwned`;
- a `GlobalLayout` with explicit rank segments, and uneven collectives ([#7670](https://github.com/NVIDIA/Megatron-LM/pull/7670), [#7671](https://github.com/NVIDIA/Megatron-LM/pull/7671));
- an owner-based compute-planning API ([#6597](https://github.com/NVIDIA/Megatron-LM/pull/6597)).

Compare it with FlexShard once it trains end to end in Megatron:
- with AdamW, against Phase C's FlexShard setup;
- with Muon against Phase B's, if its owner-based planning runs Muon.

## Roadmap

Composition with the rest of Megatron, for the Megatron DDP vs FlexShard comparison of Phases A and B, in order of benchmarking value:
1. **Tensor parallelism (Megatron vs FlexShard at TP × DP).** This is the most common Megatron configuration for dense models from about 8B up, so larger comparisons need it. `validate_args` no longer rejects it: FlexShard shards each TP rank's slices over that rank's data-parallel group (see [Design](#design)).
   - 117M model (4 layers, hidden 1024) at TP 2 × DP 2: iteration-1 loss and grad norm match Megatron exactly, with and without sequence parallelism, no-sync, `--qk-layernorm` and `--tp-comm-overlap`.
   - 1.4B model over 500 iterations at TP 2 × DP 2 and TP 2 × DP 4, with sequence parallelism: FlexShard's loss curves differ from Megatron's about as much as Megatron's two runs differ from each other.
   - 1.4B model over 500 iterations at TP 4 × DP 2, with sequence parallelism: Megatron's two runs are bit-identical here, so there is no noise floor. FlexShard matches at iteration 1, and its 50-iteration moving average of the loss stays within 0.02 of Megatron's (with and without no-sync), ending at 0.0115 vs 0.0117.
   - 7.2B model at TP 2 × DP 4 with sequence parallelism, measured with flex_shard at #21 and fusion off: FlexShard is 14% faster than Megatron at 1 microbatch (222 vs 259 ms/it, 16.8 vs 30.4 GB max allocated), 4.5% at 2 microbatches and 3% at 8 (with no-sync; 30.1 vs 30.4 GB). It doesn't need `CUDA_DEVICE_MAX_CONNECTIONS=1`: with it unset, step time stays within 2% (368 vs 366 ms/it at 2 microbatches, 1143 vs 1124 at 8).
2. **Gradient accumulation fusion.** With TransformerEngine, Megatron by default has the weight-gradient GEMM accumulate straight into an fp32 `main_grad` buffer. `--use-flex-shard` now supports it (see [Design](#design)), and on the 117M model iteration-1 loss and grad norm match Megatron exactly with fusion on both sides.
   - With activation recompute (`--recompute-modules moe`, `--recompute-granularity full`), FlexShard used to fail in the recomputed backward with no `main_grad` on TransformerEngine's grouped-tensor GEMM path (`--moe-use-grouped-tensor`). Per-expert weights on the default GEMM path, and the dense model with full recompute, already matched Megatron. The fused layers now alias `main_grad` themselves (see [Design](#design)), and the grouped-tensor path matches Megatron with MoE recompute and fusion.
   - Still to do: benchmark Megatron and FlexShard with fusion on the 1.4B and 7.2B models. The benchmarks so far turned fusion off on both sides, partly because this environment lacked APEX's `fused_weight_gradient_mlp_cuda`, which Megatron's own linear layers need for fusion; it is now built for the benchmarks.
3. **Tied embeddings.** `--use-flex-shard` now supports Megatron's default tied embedding and output weights (see [Design](#design)). On the 117M model, iteration-1 loss and grad norm match Megatron exactly at DP 4 and at TP 2 × DP 2 with sequence parallelism, with no-sync, fusion and multi-token prediction. The 1.4B model tracks Megatron over 500 iterations within its run-to-run spread.
4. **Expert parallelism (MoE, Megatron vs FlexShard).** `--use-flex-shard` now supports EP (see [Design](#design)), with flex_shard [#27](https://github.com/meta-pytorch/flex_shard/pull/27).
   - Small MoE model (4 layers, hidden 1024, 8 experts, top-2, expert FFN 2048, grouped GEMM, all-to-all dispatcher, tied embeddings) on 8 GPUs: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly at EP 1, 2, 4 and 8, and at TP 2 with sequence parallelism, expert TP 2 and 1, and EP 2 and 4. At EP 4 they also match with reshard-after-forward, no-sync, per-token loss, expert bias, shared experts with and without overlap, a dense first layer, multi-token prediction, recompute of `moe_act` and of `moe`, the allgather dispatcher, gradient accumulation fusion, experts without grouped GEMM, and with 64 experts, top-1 routing and 32-token sequences, where many experts receive no tokens.
   - 10B MoE model (Qwen3-30B-A3B layers, 16 of its 48) over 500 iterations at EP 4 × expert data-parallel 2, with no-sync over 4 microbatches: FlexShard's loss and grad-norm curves differ from Megatron's about as much as Megatron's two runs differ from each other, with no bias. On the mock data every run reaches a loss of about 0.005, so only the first 150 iterations tell the runs apart.
   - Small MoE model with TransformerEngine's grouped-tensor GEMM (`--moe-use-grouped-tensor`, per-expert weights):
     - Iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly at EP 1, 2, 4 and 8, with 4 experts at EP 1, with 64 experts and top-1 routing at EP 4, at TP 2 with sequence parallelism and expert TP 1 and 2 at EP 2, and at PP 2 × EP 2.
     - At EP 4 they also match with reshard-after-forward, syncing every microbatch, without fusion, with linear biases, MTP, MoE recompute, delayed expert weight gradients, the EP overlap with and without delayed weight gradients, and blockwise FP8 parameter all-gather.
     - Checkpoints at EP 4: resuming FlexShard at iteration 5 stays within 1.1e-3 of an uninterrupted run, about as far as two MoE runs drift apart. Loading Megatron's checkpoint into FlexShard reproduces Megatron's next loss exactly.
   - Still to do: benchmark Megatron and FlexShard on the 10B MoE model.
5. **Pipeline parallelism.** `--use-flex-shard` now supports PP and virtual PP (see [Design](#design)).
   - 117M model with tied embeddings and 8 microbatches: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly at PP 2 × DP 4, PP 4 × DP 2, PP 2 × TP 2 × DP 2 with sequence parallelism, and PP 2 with 2 virtual pipeline chunks, for FlexShard with no-sync, with reshard-after-forward, and syncing every microbatch. At PP 2 × DP 4 they also match with MTP, per-token loss, full recompute, an uneven layer split, fusion and untied embeddings, and so does the small MoE model at PP 2 × EP 2, with and without TP 2.
   - 1.4B model over 500 iterations at PP 2 × DP 4 and PP 4 × DP 2: FlexShard's loss curves differ from Megatron's about as much as Megatron's two runs differ from each other.
   - Later stages reduce-scatter in the pipeline bubble, as Megatron DDP does, through flex_shard's `finalize_backward` ([#32](https://github.com/meta-pytorch/flex_shard/pull/32)). Iteration-1 loss and grad norm still match Megatron exactly at PP 2 × DP 2, with and without 2 virtual pipeline chunks and with MTP, and for the small MoE model at PP 2 × EP 2.
   - Still to do: a benchmark, with and without the overlap in the bubble.
6. **Context parallelism.** `--use-flex-shard` composes with CP without changes (see [Design](#design)).
   - 117M model with tied embeddings at an 8K sequence and 8 microbatches: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly at CP 2 × DP 4, with and without reshard-after-forward and with no-sync.
   - Still to do: CP 2 × TP 2 × DP 2 and CP 2 × PP 2 × DP 2; CP 4 and 8; the all-gather, all-to-all and hierarchical CP communication types; MTP, recompute, fusion, per-token loss and untied embeddings; the small MoE model with EP; loss curves on the 1.4B model; and a long-context benchmark.
7. **FP8 parameter all-gather.** `--use-flex-shard` all-gathers TransformerEngine's weights in FP8 with `--fp8-param-gather --fp8-recipe blockwise` (see [Design](#design)); with other recipes it still switches the flag off with a warning.
   - 117M model with the blockwise recipe at DP 8: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer's blockwise FP8 training without `--fp8-param-gather` exactly, and peak allocated memory drops by 94 MB (1224 vs 1317 MB) against FlexShard's bf16 all-gather.
   - Composition matrix, against the same Megatron baseline: the iteration-1 loss and grad norm and the iteration-2 loss match exactly on the 117M model with TP 2, PP 2, per-token loss, full recompute and gradient accumulation fusion, and on the small MoE model with EP 2 (with and without the EP overlap) and with grouped-tensor experts at EP 4.
     - Later iterations differ by at most 5.4e-4 on the 117M model, against 4.9e-4 at plain DP 8.
     - On the MoE model they differ by 1.8e-3 to 3.2e-3; two Megatron runs differ by 1.2e-3 to 1.3e-3 there.
   - Still to do: loss curves against Megatron with `--fp8-param-gather`, and a benchmark.
8. **Single grouped MoE weights.** `--use-flex-shard` supports `--moe-single-grouped-weight` and `--moe-single-grouped-bias` for bf16 weights (see [Design](#design)).
   - These runs used a TransformerEngine 2.21.0.dev0 rebuilt against cuBLAS 13.8, preloaded at run time, since this node's CUDA 13.0 and 13.1 ship cuBLAS 13.1 and 13.2.
   - A standalone check on one `GroupedLinear`: forward output, input gradient, and weight and bias gradients are bit-identical to TransformerEngine's own single grouped parameters, with and without fusion, with delayed weight gradients, with two microbatches interleaved, and after the gathered storage is freed and refilled.
   - Small MoE model at EP 4 with no-sync, `--moe-grouped-gemm --moe-use-grouped-tensor --moe-single-grouped-weight` on both sides: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly, and max allocated memory matches FlexShard with per-expert weights (1655.0 vs 1656.0 MB).
   - Composition matrix on the small MoE model:
     - Iteration-1 loss and grad norm match exactly at EP 1, 2 and 8, with 4 experts at EP 1, with 64 experts and top-1 routing at EP 4, at TP 2 with sequence parallelism and expert TP 1 and 2 at EP 2, and at PP 2 × EP 2.
     - At EP 4 they also match with reshard-after-forward, syncing every microbatch, without fusion, with single grouped biases, MTP, MoE recompute, delayed expert weight gradients, and the EP overlap with and without delayed weight gradients.
   - Checkpoints at EP 4: resuming FlexShard at iteration 5 stays within 1.3e-3 of an uninterrupted run. Megatron's and FlexShard's checkpoints load into each other, reproducing the next loss exactly.
   - Still to do: loss curves on the 10B MoE model, and a benchmark against per-expert weights.
9. **Real training runs.**
   - **Distributed checkpoint save/load** in `torch_dist` (see [Design](#design)). A unit test of the composition passes: for a TP-split weight with a layer axis and for SwiGLU's gate/up factory, cut into uneven row chunks over 3 data-parallel ranks, the pieces tile the region exactly once at the right global offsets, view the local shard and merge back exactly.
     - 117M model at DP 8: resuming at iteration 5 stays within 1.1e-4 of an uninterrupted run over iterations 6–10, with and without reshard-after-forward, against 8.7e-5 for Megatron's own resume.
     - Loading Megatron's checkpoint into FlexShard, and FlexShard's into Megatron, reproduces the next loss exactly (weights only, `--no-load-optim`).
     - The small MoE model at EP 4 also resumes and cross-loads, with grouped-tensor experts (item 4) and with single grouped weights (item 8).
     - Still to do (`FLEXSHARD_CHECKPOINT_PLAN.md`): loading at a different DP size, TP and PP, and a mid-run resume on the 1.4B model.
   - **Evaluation** (`--eval-iters > 0`) is untested, since every benchmark ran with evaluation off.
