# Accelerate Internals: Level 6 — Fully Sharded Data Parallel (FSDP)

> **Series:** Contributing to HuggingFace Accelerate — A Developer's Field Guide
> **Chapter:** 6 — Fully Sharded Data Parallel
> **Primary source files:**
> `src/accelerate/utils/dataclasses.py` — `FullyShardedDataParallelPlugin`
> `src/accelerate/utils/fsdp_utils.py` — wrapping policies, state dict helpers
> `src/accelerate/accelerator.py` — `prepare_model()` FSDP branch
> `examples/by_feature/fsdp_with_peak_mem_tracking.py`
> `tests/fsdp/test_fsdp.py`

---

## Table of Contents

1. [Why FSDP Exists: The Memory Wall](#1-why-fsdp-exists-the-memory-wall)
2. [The Memory Arithmetic](#2-the-memory-arithmetic)
3. [FSDP Communication Primitives](#3-fsdp-communication-primitives)
4. [Source Walk: FullyShardedDataParallelPlugin](#4-source-walk-fullyshardeddataparallelplugin)
5. [Source Walk: prepare() FSDP Branch](#5-source-walk-prepare-fsdp-branch)
6. [Sharding Strategies In Depth](#6-sharding-strategies-in-depth)
7. [Wrapping Policies: What Gets Sharded and Why](#7-wrapping-policies-what-gets-sharded-and-why)
8. [Mixed Precision with FSDP](#8-mixed-precision-with-fsdp)
9. [Checkpointing: FULL vs SHARDED State Dicts](#9-checkpointing-full-vs-sharded-state-dicts)
10. [FSDP1 vs FSDP2: Architecture Deep Dive](#10-fsdp1-vs-fsdp2-architecture-deep-dive)
11. [FSDP2 in Accelerate: Source-Level Details](#11-fsdp2-in-accelerate-source-level-details)
12. [FSDP2 Mixed Precision and FP8](#12-fsdp2-mixed-precision-and-fp8)
13. [FSDP2 Checkpointing with DTensor](#13-fsdp2-checkpointing-with-dtensor)
14. [CPU RAM-Efficient Loading](#14-cpu-ram-efficient-loading)
15. [Activation Checkpointing with FSDP](#15-activation-checkpointing-with-fsdp)
16. [Gradient Accumulation with FSDP](#16-gradient-accumulation-with-fsdp)
17. [Complete Training Example](#17-complete-training-example)
18. [Complete FSDP2 Training Example](#18-complete-fsdp2-training-example)
19. [Contribution Opportunities](#19-contribution-opportunities)
20. [What You Should Know After This Chapter](#20-what-you-should-know-after-this-chapter)

---

## 1. Why FSDP Exists: The Memory Wall

DDP, which we covered in Level 2, gives every GPU a complete, independent replica of the model. That is fast. Each process runs a full forward and backward pass locally, synchronizes only gradients over a ring all-reduce, and never needs to coordinate anything else. The replica model is cheap communication-wise, and as long as your model fits on a single GPU, DDP is the right default.

The moment your model stops fitting on a single GPU, DDP breaks entirely. You cannot even instantiate the model, let alone wrap it.

This is the problem FSDP solves. Instead of replicating, it **shards**: every GPU owns only a fraction of every layer's parameters, every layer's gradients, and every parameter's optimizer state. When a layer needs to compute, the GPUs cooperate to reconstruct that layer momentarily, compute through it, then discard the reconstructed copy and keep only their local shard. The memory footprint per GPU scales as `1/num_gpus` for all three memory-heavy objects (parameters, gradients, optimizer state), which is what makes training multi-billion-parameter models feasible without specialized hardware.

FSDP in PyTorch is directly descended from Microsoft's ZeRO-3 (Zero Redundancy Optimizer, Stage 3), which was introduced in the DeepSpeed library. The key insight of ZeRO-3 is that the three largest memory consumers in a standard training run (parameters, gradients, optimizer state) need not all live on every GPU simultaneously. They can be distributed and temporarily gathered only when needed.

---

## 2. The Memory Arithmetic

Before reading any code, you need to internalize the memory math. All calculations assume 32-bit floats (4 bytes each) unless noted.

### DDP on a Single GPU (1B parameter model)

| Component | Size | Formula |
|---|---|---|
| Parameters (fp32) | 4 GB | `1e9 params × 4 bytes` |
| Gradients (fp32) | 4 GB | `1e9 params × 4 bytes` |
| Adam optimizer state | 8 GB | `momentum + variance × 4 bytes each` |
| Activations (varies) | 2-10 GB | depends on batch size and sequence length |
| **Total per GPU** | **~18 GB minimum** | excludes activations |

Every GPU in a DDP job holds all 18 GB. Adding a second GPU does not reduce memory per GPU. It just doubles your throughput.

### FSDP on 4 GPUs (same 1B parameter model, FULL_SHARD)

| Component | Size per GPU | Formula |
|---|---|---|
| Parameters (sharded) | 1 GB | `4 GB / 4 GPUs` |
| Gradients (sharded) | 1 GB | `4 GB / 4 GPUs` |
| Adam optimizer state (sharded) | 2 GB | `8 GB / 4 GPUs` |
| Unsharded working copy (peak) | 4 GB | temporarily gathered for one layer at a time |
| **Total per GPU** | **~8 GB** | down from 18 GB |

The "unsharded working copy" is the cost of the all-gather operation before each layer's forward pass. FSDP gathers all shards for one layer, computes through it, then discards the gathered copy. The peak memory usage therefore depends on the size of your largest FSDP-wrapped unit, not the size of the entire model. This is why wrapping policy matters so much: wrapping at the transformer-block granularity means the largest gather is one block, not the whole model.

### The Scaling Law

With `N` GPUs in FULL_SHARD mode:

```
memory_per_gpu = (total_params + total_grads + total_optimizer_state) / N
                 + peak_activation_memory_for_largest_fsdp_unit
```

The activation term does not scale with N because each GPU still processes a full micro-batch. However, with mixed precision (bf16 parameters, fp32 gradient accumulation), the activation footprint is also halved compared to a pure fp32 training run.

---

## 3. FSDP Communication Primitives

FSDP uses two collective operations from NCCL that are distinct from the all-reduce used by DDP. Understanding them is essential before reading the source code.

### All-Gather

An all-gather collects one tensor shard from each rank and broadcasts the complete concatenated tensor to every rank. The result is that every rank ends up with the full tensor.

```
Before all-gather (4 GPUs, each has shard of param W):
  GPU0: W[0:256]
  GPU1: W[256:512]
  GPU2: W[512:768]
  GPU3: W[768:1024]

After all-gather:
  GPU0: W[0:1024]  <-- full parameter
  GPU1: W[0:1024]  <-- full parameter
  GPU2: W[0:1024]  <-- full parameter
  GPU3: W[0:1024]  <-- full parameter
```

FSDP runs an all-gather before each layer's forward pass so every rank has the full weights it needs to compute.

### Reduce-Scatter

A reduce-scatter reduces (sums) a tensor across all ranks, then distributes the result so each rank receives only its fraction.

```
Before reduce-scatter (4 GPUs, each has full gradient dW from backward):
  GPU0: dW_full_from_gpu0
  GPU1: dW_full_from_gpu1
  GPU2: dW_full_from_gpu2
  GPU3: dW_full_from_gpu3

After reduce-scatter:
  GPU0: sum(dW)[0:256]    <-- owns only shard 0 of the reduced gradient
  GPU1: sum(dW)[256:512]
  GPU2: sum(dW)[512:768]
  GPU3: sum(dW)[768:1024]
```

FSDP runs a reduce-scatter after backward to synchronize and re-shard gradients. Each rank then applies the optimizer update to only its own shard.

### The Full FSDP Step

```
Forward pass (per layer):
  1. all-gather parameters for this layer
  2. run forward computation
  3. discard non-local parameter shards (free memory)

Backward pass (per layer, in reverse):
  4. all-gather parameters for this layer (needed for gradient computation)
  5. run backward computation (compute local gradients)
  6. reduce-scatter gradients (sum across ranks, each keeps its shard)
  7. discard non-local parameter shards

Optimizer step:
  8. apply optimizer update to local parameter shard only
```

Steps 1 and 4 (the all-gathers) are the primary communication overhead. FSDP's `backward_prefetch` feature overlaps step 4 for layer `N-1` with step 5 for layer `N`, hiding communication latency behind compute.

---

## 4. Source Walk: FullyShardedDataParallelPlugin

**File:** `src/accelerate/utils/dataclasses.py`

`FullyShardedDataParallelPlugin` is the data class that collects every FSDP configuration knob and translates it into arguments suitable for passing to PyTorch's `FullyShardedDataParallel` constructor (FSDP1) or to `torch.distributed.fsdp.fully_shard` (FSDP2). It is not a plugin in the sense that it modifies Accelerate's internal dispatch: it is just a structured bag of configuration that `Accelerator` inspects when its `distributed_type` is `FSDP`.

The class signature (simplified) is:

```python
@dataclass
class FullyShardedDataParallelPlugin:
    # --- Version selector ---
    fsdp_version: int = 1
    # 1 = legacy FullyShardedDataParallel class (FSDP1)
    # 2 = fully_shard() function API (FSDP2, DTensor-based)

    # --- Sharding strategy (FSDP1) / resharding flag (FSDP2) ---
    sharding_strategy: Optional[Union[str, ShardingStrategy]] = None
    # Deprecated in favor of reshard_after_forward.
    reshard_after_forward: Union[str, ShardingStrategy, bool] = "FULL_SHARD"
    # In FSDP1: mirrors ShardingStrategy (FULL_SHARD, SHARD_GRAD_OP, etc.)
    # In FSDP2: bool (True = FULL_SHARD behavior, False = SHARD_GRAD_OP behavior)

    # --- Communication tuning ---
    backward_prefetch: Optional[Union[str, BackwardPrefetch]] = "NO_PREFETCH"
    # BACKWARD_PRE: prefetch next layer's all-gather before current backward
    # BACKWARD_POST: prefetch after current backward (less overlap)
    # NO_PREFETCH: no prefetching (safest but slowest)

    forward_prefetch: bool = False
    # Whether to prefetch next layer's all-gather before current forward
    # Useful for compute-bound workloads

    # --- Wrapping ---
    auto_wrap_policy: Optional[Union[Callable, str]] = None
    # Which modules get wrapped in FSDP units

    # --- Precision and offload ---
    mixed_precision_policy: Optional[Union[dict, str, MixedPrecision]] = None
    cpu_offload: Union[bool, CPUOffload] = False

    # --- State dict type ---
    state_dict_type: Optional[Union[str, StateDictType]] = "FULL_STATE_DICT"

    # --- Model loading ---
    sync_module_states: bool = True
    cpu_ram_efficient_loading: bool = False

    # --- FSDP2-only: FP8 all-gather ---
    enable_fsdp_float8_all_gather: Optional[bool] = None
    # When True, FSDP2 casts parameters to FP8 before the all-gather,
    # saving 50% all-gather bandwidth compared to BF16.
```

### The `__post_init__` Normalization Pass

After the dataclass fields are set, `__post_init__` does significant work to normalize inputs. It converts string sharding strategy names (`"FULL_SHARD"`, `"SHARD_GRAD_OP"`) to the actual `ShardingStrategy` enum values. It also handles the case where the deprecated `sharding_strategy` field was set instead of the new `reshard_after_forward`, ensuring backward compatibility.

The key code (paraphrased) is roughly:

```python
def __post_init__(self):
    # Resolve string -> enum for sharding strategy
    if isinstance(self.reshard_after_forward, str):
        if self.fsdp_version == 2:
            # FSDP2 only understands bool
            self.reshard_after_forward = (
                self.reshard_after_forward.upper() != "SHARD_GRAD_OP"
            )
        else:
            # FSDP1: convert string to ShardingStrategy enum
            self.reshard_after_forward = ShardingStrategy[
                self.reshard_after_forward.upper()
            ]

    # Resolve backward_prefetch string -> enum
    if isinstance(self.backward_prefetch, str):
        self.backward_prefetch = BackwardPrefetch[
            self.backward_prefetch.upper()
        ]

    # Resolve state_dict_type string -> enum
    if isinstance(self.state_dict_type, str):
        self.state_dict_type = StateDictType[self.state_dict_type.upper()]

    # Resolve auto_wrap_policy string -> callable
    if isinstance(self.auto_wrap_policy, str):
        self.auto_wrap_policy = ...  # maps string to partial function
```

This normalization is what allows YAML config values like `fsdp_sharding_strategy: FULL_SHARD` to map cleanly to the Python enum, without the user ever touching the enum directly.

### The `set_auto_wrap_policy` Method

This method handles the three recognized policy strings:

```python
def set_auto_wrap_policy(self, model):
    if self.auto_wrap_policy == "TRANSFORMER_BASED_WRAP":
        # Reads fsdp_transformer_layer_cls_to_wrap from config,
        # builds functools.partial(transformer_auto_wrap_policy, ...)
        ...
    elif self.auto_wrap_policy == "SIZE_BASED_WRAP":
        # Uses default_auto_wrap_policy with min_num_params threshold
        ...
    elif self.auto_wrap_policy == "NO_WRAP":
        self.auto_wrap_policy = None
```

The transformer-based policy is the one you will use for almost every real workload. It wraps each specified layer class (e.g., `LlamaDecoderLayer`) in its own FSDP unit. The result is that memory usage per FSDP unit equals roughly the memory of a single transformer block, which is manageable even for very large models.

---

## 5. Source Walk: prepare() FSDP Branch

**File:** `src/accelerate/accelerator.py`, `prepare_model()` method

When `prepare()` is called with a model and `accelerator.state.distributed_type == DistributedType.FSDP`, it routes to `prepare_model()`, which branches on the FSDP version.

Here is the annotated control flow:

```python
def prepare_model(self, model, device_placement=None, evaluation_mode=False):
    # ...
    if self.distributed_type == DistributedType.FSDP:
        # 1. Check that the model hasn't already been FSDP-wrapped
        #    (avoids the double-prepare bug we found in Level 1).
        if type(model) in (FullyShardedDataParallel,):
            return model  # already wrapped, no-op

        # 2. Determine the FSDP version
        fsdp_plugin = self.state.fsdp_plugin

        if fsdp_plugin.fsdp_version == 2:
            # 3a. FSDP2 path: uses fully_shard() function API
            model = self._prepare_fsdp2_model(model)
        else:
            # 3b. FSDP1 path: wraps with FullyShardedDataParallel class
            model = self._prepare_fsdp1_model(model)

        # 4. Register the wrapped model
        self._models.append(model)
        return model
```

### FSDP1 Model Preparation

The FSDP1 path (`_prepare_fsdp1_model`, simplified):

```python
def _prepare_fsdp1_model(self, model):
    kwargs = {
        "sharding_strategy": fsdp_plugin.reshard_after_forward,
        "cpu_offload": CPUOffload(offload_params=True)
            if fsdp_plugin.cpu_offload else None,
        "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
        "mixed_precision": fsdp_plugin.mixed_precision_policy,
        "sync_module_states": fsdp_plugin.sync_module_states,
        "device_id": torch.cuda.current_device(),
        "use_orig_params": fsdp_plugin.use_orig_params,
        "backward_prefetch": fsdp_plugin.backward_prefetch,
        "forward_prefetch": fsdp_plugin.forward_prefetch,
        "ignored_modules": fsdp_plugin.ignored_modules,
        "limit_all_gathers": fsdp_plugin.limit_all_gathers,
    }

    model = FullyShardedDataParallel(model, **kwargs)
    return model
```

The `device_id=torch.cuda.current_device()` argument is critical. It tells FSDP1 to immediately move the shards to the current GPU rather than remaining on CPU. Without this, the model shards stay on CPU memory and FSDP still works but is dramatically slower.

The `use_orig_params=True` argument is also important in practice. Without it, FSDP1 flattens all module parameters into a single `FlatParameter`, which makes optimizer parameter groups break (you can no longer apply different learning rates to different layers). With `use_orig_params=True`, the original parameter names and shapes are preserved in the Python object, though internally FSDP still manages them as flattened shards.

### FSDP2 Model Preparation

The FSDP2 path is fundamentally different because it does not use the `FullyShardedDataParallel` wrapper class at all. Instead, it applies `fully_shard()` as an in-place mutation of the model:

```python
def _prepare_fsdp2_model(self, model, optimizer=None):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    # Build the FSDP2 mesh (1D data-parallel mesh by default)
    mesh = init_device_mesh("cuda", (self.num_processes,))

    # Build kwargs for fully_shard
    fsdp_kwargs = {}
    if fsdp_plugin.mixed_precision_policy is not None:
        fsdp_kwargs["mp_policy"] = fsdp_plugin.mixed_precision_policy

    # Apply fully_shard to each specified sub-module first (bottom-up)
    # This is the FSDP2 equivalent of auto_wrap_policy
    for module_name, module in model.named_modules():
        if _should_wrap_module(module, fsdp_plugin.auto_wrap_policy):
            fully_shard(module, mesh=mesh, **fsdp_kwargs)

    # Apply to the root model last
    fully_shard(model, mesh=mesh, **fsdp_kwargs)

    # FSDP2 REQUIRES the optimizer to be created AFTER fully_shard
    # because fully_shard converts parameters to DTensor in-place,
    # and an optimizer built before that would hold references to
    # the original torch.Tensor parameters, not the DTensor shards.
    if optimizer is not None:
        optimizer.param_groups = ...  # re-bind to DTensor parameters

    return model
```

This is why FSDP2 in Accelerate has a hard constraint visible in the source: when `prepare()` is called with `fsdp_version=2`, the model and optimizer must be passed together in the same `prepare()` call. Accelerate raises `ValueError` if you try to prepare them separately, because it needs to re-initialize the optimizer after `fully_shard()` converts the parameters to `DTensor`.

---

## 6. Sharding Strategies In Depth

FSDP1 offers five sharding strategies, numbered 1 through 5 in the config YAML. FSDP2 simplifies this to a boolean. Understanding when to use each strategy is one of the most practically important things in this chapter.

### FULL_SHARD (Strategy 1, FSDP2: `reshard_after_forward=True`)

**What is sharded:** parameters + gradients + optimizer state.

**Communication pattern:** all-gather before forward, reduce-scatter after backward. After the forward pass, the gathered parameters are discarded immediately.

**Memory:** lowest possible. Each GPU holds approximately `total_model_memory / N`.

**Communication:** highest. Two collective operations per layer per step (all-gather before forward, all-gather before backward, reduce-scatter after backward). The second all-gather (before backward) is what makes this mode expensive.

**When to use:** when the model barely fits or does not fit without sharding. The additional communication is the price you pay for memory.

```python
from torch.distributed.fsdp import ShardingStrategy

plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # FSDP1
    # or for FSDP2:
    fsdp_version=2,
    reshard_after_forward=True,
)
```

### SHARD_GRAD_OP (Strategy 2, FSDP2: `reshard_after_forward=False`)

**What is sharded:** gradients + optimizer state. Parameters are sharded at rest but kept unsharded (not freed) after the forward pass.

**Communication pattern:** all-gather before forward only. After forward, the full parameter copy is retained for the backward pass. After backward, reduce-scatter for gradients. After optimizer step, parameters are re-sharded.

**Memory:** higher than FULL_SHARD. The full parameter tensor lives in GPU memory throughout the backward pass.

**Communication:** lower. Only one all-gather per layer per step (before forward). The backward pass reads from the already-gathered copy.

**When to use:** when the model fits in memory with the full parameters kept but you still want sharded optimizer state to reduce memory by 2x from DDP.

```python
plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # FSDP1
    # or for FSDP2:
    fsdp_version=2,
    reshard_after_forward=False,
)
```

### HYBRID_SHARD (Strategy 4)

**What is sharded:** everything, but only within a node. Across nodes, each node holds a full replica.

**Communication pattern:** all-gather and reduce-scatter happen intra-node over NVLink (fast). Cross-node communication is an all-reduce over the slow inter-node link (Ethernet or InfiniBand), exactly as in DDP.

**When to use:** multi-node training when inter-node bandwidth is the bottleneck. This is the standard configuration for large-scale training on clusters where intra-node NVLink is 600 GB/s but inter-node bandwidth is 200 GB/s.

```python
plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
)
```

### NO_SHARD (Strategy 3)

This is equivalent to DDP. All parameters, gradients, and optimizer states are fully replicated. FSDP with NO_SHARD is functionally identical to `DistributedDataParallel` but is useful when you want to use the same FSDP codepath without the memory savings, for example when benchmarking or debugging.

### HYBRID_SHARD_ZERO2 (Strategy 5)

Like HYBRID_SHARD but uses SHARD_GRAD_OP semantics within each node. Optimizer states and gradients are sharded, but parameters are not discarded after the forward pass within the node. A niche option for specific hardware and memory configurations.

---

## 7. Wrapping Policies: What Gets Sharded and Why

The wrapping policy answers the question: which `nn.Module` instances get their own FSDP unit? This matters because the unit of sharding is the module, not the individual parameter. When FSDP all-gathers parameters for a unit, it must gather the entire unit at once, so the peak memory usage during a step is determined by the largest single FSDP unit.

### Why You Cannot Shard Mid-Attention

A single attention layer (multi-head self-attention) contains the Q, K, V projection weights and the output projection. These four parameter tensors are used together in a single forward computation, and the backward computation of any one of them requires all four to be present. If you wrap Q, K, V, and O in separate FSDP units, FSDP would need to all-gather each one independently, but the backward pass of Q requires O's gradient, which means O must still be in memory when Q's backward runs. This creates a correctness problem.

The safe wrapping boundary is the **transformer block**: one FSDP unit per decoder or encoder block. Within a block, the attention and MLP sublayers are tightly coupled, so they must be wrapped together.

### Transformer-Based Wrapping (TRANSFORMER_BASED_WRAP)

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# Manually specify which module class defines a block boundary
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

plugin = FullyShardedDataParallelPlugin(
    auto_wrap_policy=auto_wrap_policy,
)
```

Accelerate's YAML equivalent:

```yaml
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

You can specify multiple layer classes as a comma-separated list, which is important when your model has heterogeneous block types:

```yaml
fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer,LlamaEmbedding
```

The key constraint from the Accelerate docs: modules that share weights (like embedding layers tied to the language model head) must not end up in different FSDP units. If they do, you will get an error when FSDP tries to shard them because sharding a parameter that already appears in another FSDP unit's list is undefined behavior. Transformer-based wrapping is designed to put the shared embedding layer in the outermost FSDP unit (the root), keeping it together with the LM head.

### Size-Based Wrapping (SIZE_BASED_WRAP)

```yaml
fsdp_auto_wrap_policy: SIZE_BASED_WRAP
fsdp_min_num_params: 100000
```

Any `nn.Module` with more than `fsdp_min_num_params` parameters gets wrapped. This is simpler but less semantically correct for transformer models, because it may wrap sublayers that should stay together.

### Custom Wrapping Policy

For fine-grained control, you can pass any callable that takes a module and returns a boolean:

```python
def my_wrap_policy(module, recurse, nonwrapped_numel):
    # recurse=True means: should we recurse into children?
    # recurse=False means: should this module itself be wrapped?
    if recurse:
        return True  # always recurse to children
    # Wrap if this is a transformer decoder block
    return isinstance(module, (MyDecoderLayer, MyCrossAttentionLayer))

plugin = FullyShardedDataParallelPlugin(
    auto_wrap_policy=my_wrap_policy,
)
```

### Verifying Your Wrapping Structure

After wrapping, print the model to see the FSDP unit boundaries:

```python
model = accelerator.prepare(model)
# unwrap_model gives access to the FSDP wrapper
unwrapped = accelerator.unwrap_model(model)
print(model)  # shows which modules are FullyShardedDataParallel instances
```

In the output, each `FullyShardedDataParallel` wrapper indicates a sharding boundary. You want to see one per transformer block, not one per linear layer (too many small units, too much communication overhead) and not one for the entire model (too large a working set in memory during the step).

---

## 8. Mixed Precision with FSDP

FSDP has its own mixed-precision mechanism that is separate from Accelerate's top-level `mixed_precision` setting. The key distinction: Accelerate's top-level mixed precision controls `autocast()` (which operation computes in fp16/bf16), while FSDP's `mixed_precision_policy` controls **what dtype the parameters are stored and communicated in**.

```python
from torch.distributed.fsdp import MixedPrecision
import torch

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,    # weights are stored and all-gathered in bf16
    reduce_dtype=torch.bfloat16,   # gradients are reduced in bf16
    buffer_dtype=torch.bfloat16,   # buffers (BatchNorm running_mean, etc.) in bf16
)

plugin = FullyShardedDataParallelPlugin(
    mixed_precision_policy=mixed_precision_policy,
)
```

The `param_dtype` choice directly affects the size of the all-gather communication. If `param_dtype=bfloat16`, the all-gather transfers 2 bytes per parameter instead of 4, halving communication volume. This is a significant practical speedup on bandwidth-limited hardware.

The `reduce_dtype` is separate from `param_dtype` because reducing gradients in fp16 or bf16 can introduce meaningful numerical errors at large scale. Some users keep `param_dtype=bfloat16` for communication efficiency but set `reduce_dtype=float32` for numerical stability:

```python
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # gradient reduction in fp32 for stability
    buffer_dtype=torch.bfloat16,
)
```

### YAML Configuration

```yaml
mixed_precision: bf16
# This sets both Accelerate-level autocast AND FSDP param/reduce dtype
# to bfloat16. Under the hood Accelerate builds a MixedPrecision object.
```

Alternatively, you can set `mixed_precision: no` in YAML and pass `mixed_precision_policy` directly to the plugin for maximum control.

---

## 9. Checkpointing: FULL vs SHARDED State Dicts

Checkpointing with FSDP is where many users hit painful bugs. The difficulty is that `model.state_dict()` behavior changes completely depending on which `StateDictType` is active. Getting this wrong produces either OOM errors, silently incomplete saves, or checkpoint files that cannot be loaded back without all the original ranks.

### FULL_STATE_DICT: Gather to Rank 0

In `FULL_STATE_DICT` mode, calling `model.state_dict()` triggers an all-gather: every shard is sent to rank 0, which assembles the complete parameter tensors. Rank 0 then saves a single checkpoint file that looks exactly like a non-FSDP checkpoint.

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import torch.distributed.fsdp as fsdp_module

# Configure state dict gathering
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with fsdp_module.FullyShardedDataParallel.state_dict_type(
    model, StateDictType.FULL_STATE_DICT, save_policy
):
    state_dict = model.state_dict()

# Only rank 0 saves
if accelerator.is_main_process:
    torch.save(state_dict, "checkpoint.pt")
```

With `offload_to_cpu=True`, gathered tensors are immediately moved to CPU to avoid GPU OOM. With `rank0_only=True`, only rank 0 receives the full state dict; other ranks receive empty dicts, which avoids the `N×model_size` memory spike that would occur if every rank received the full gathered state.

**The limitation:** even with `offload_to_cpu=True`, you need enough CPU RAM to hold the full model. For a 70B model in bf16, that is 140 GB of RAM. This is feasible on a well-provisioned machine but impossible on a cloud instance with limited RAM. It also induces significant latency from the all-gather, which can trigger NCCL timeouts on long checkpointing intervals.

### SHARDED_STATE_DICT: Each Rank Saves Its Own Shard

In `SHARDED_STATE_DICT` mode, each rank saves only its own shard. No inter-rank communication is needed at save time.

```python
# Accelerate's built-in save/load state uses SHARDED_STATE_DICT
# when configured in the plugin
plugin = FullyShardedDataParallelPlugin(
    state_dict_type="SHARDED_STATE_DICT",
)
```

When you call `accelerator.save_state("my_checkpoint/")`, Accelerate calls:

```python
# Each rank saves its shard independently
# Result: my_checkpoint/pytorch_model_0/__0_0.distcp __1_0.distcp ...
```

The resulting files use the `.distcp` format (Distributed Checkpoint), which is PyTorch's native sharded checkpoint format backed by `torch.distributed.checkpoint` (DCP). These files are NOT loadable as a regular `torch.load()` checkpoint. To load them, you must use `dcp.load()` with the same number of ranks, or use Accelerate's `load_state()` which handles the DCP internals.

The advantages of SHARDED_STATE_DICT are: no communication overhead during save, no CPU RAM spike, and faster saves proportional to `1/N` since each rank writes less data. This is the **recommended format** for intermediate checkpoints during training.

### Merging Sharded Checkpoints

To convert a sharded checkpoint into a single loadable file after training:

```bash
# Accelerate's CLI tool
accelerate merge-weights ./my_checkpoint/pytorch_model_0/ ./merged_model.safetensors
```

Or programmatically using Accelerate's `merge_weights_of_fsdp_checkpoints`:

```python
from accelerate.utils import merge_fsdp_weights

merge_fsdp_weights(
    checkpoint_dir="./my_checkpoint/pytorch_model_0/",
    output_path="./merged.safetensors",
    safe_serialization=True,
)
```

### Saving via Accelerate's High-Level API

The recommended pattern for saving and loading with FSDP in Accelerate is to use `save_state` and `load_state`, which handle all the StateDictType context manager setup internally:

```python
# Save (handles FSDP state dict context internally)
accelerator.save_state("./checkpoint_dir/")

# Load (re-shards across however many ranks are active)
accelerator.load_state("./checkpoint_dir/")
```

For HuggingFace model saving via `save_pretrained`, you must pass `get_state_dict` explicitly:

```python
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    args.output_dir,
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(model),  # critical line
)
```

`accelerator.get_state_dict(model)` uses `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` internally. Without this argument, `save_pretrained` calls `model.state_dict()` without any FSDP context manager, which returns shard-local tensors with FlatParameter-derived names that cannot be loaded into a vanilla HuggingFace model.

---

## 10. FSDP1 vs FSDP2: Architecture Deep Dive

This section is the heart of the chapter. FSDP2 is the current direction of the project, and as a contributor, you will be working primarily with FSDP2 code. Understanding the architectural difference between FSDP1 and FSDP2 at the implementation level is what separates a contributor from a user.

### FSDP1: The FlatParameter Architecture

In FSDP1, when you wrap a module, FSDP immediately flattens all of its parameters into a single 1D tensor called `FlatParameter`. If a transformer block has 10 parameter tensors (Q, K, V, O projections, two MLP layers, two layer-norm weights, two layer-norm biases), FSDP1 concatenates them all into one `FlatParameter` of combined size, then shards that 1D tensor across ranks.

The result looks like this at the Python level:

```
Before FSDP1 wrapping:
  layer.self_attn.q_proj.weight  shape: [4096, 4096]
  layer.self_attn.k_proj.weight  shape: [4096, 4096]
  layer.self_attn.v_proj.weight  shape: [4096, 4096]
  ...

After FSDP1 wrapping (on 4 GPUs):
  layer._flat_param               shape: [total_numel / 4]  # local shard
  # Original parameter names are no longer real tensors;
  # they are views computed on-the-fly from the FlatParameter
```

The `FlatParameter` approach has several consequences:

**Consequence 1: Metadata loss.** All original per-parameter attributes (dtype, `requires_grad`, gradient hooks) are combined into the FlatParameter. Storing different dtypes for different parameters is technically possible but requires ugly workarounds. FSDP1 with LoRA (where adapter weights should stay in fp32 while the frozen base weights are in bf16) is a known pain point.

**Consequence 2: Complex state dicts.** The state dict produced by `model.state_dict()` on an FSDP1 model contains a key `_flat_param` unless `use_orig_params=True` is set. With `use_orig_params=True`, FSDP1 tracks metadata to reconstruct the original parameter names, making the state dict look normal, but the reconstruction logic is complex and was a recurring source of bugs.

**Consequence 3: Frozen parameter complications.** If some parameters are frozen (`requires_grad=False`), they are still included in the FlatParameter. LoRA-style fine-tuning where 90% of parameters are frozen does not save communication: FSDP1 all-gathers the full FlatParameter including frozen weights, because it cannot shard a subset of a concatenated tensor.

**Consequence 4: Memory non-determinism.** FSDP1 uses `tensor.record_stream()` to manage multi-stream memory coordination (communication and compute happen in different CUDA streams). `record_stream` tells the CUDA caching allocator not to reuse a tensor's memory until a specific stream has finished. The implementation of `record_stream` in the CUDA caching allocator can cause memory fragmentation and non-deterministic peak memory usage.

### FSDP2: The DTensor Architecture

FSDP2 abandons `FlatParameter` entirely. Each parameter is sharded individually as a `DTensor` (Distributed Tensor). `DTensor` is a first-class PyTorch primitive that represents a tensor partitioned across a `DeviceMesh` with explicit placement semantics.

After `fully_shard(model)`:

```
Before FSDP2 sharding:
  layer.self_attn.q_proj.weight  type: torch.Tensor, shape: [4096, 4096]

After FSDP2 sharding (on 4 GPUs):
  layer.self_attn.q_proj.weight  type: DTensor, shape: [4096, 4096]
                                  # logical shape is unchanged
                                  # .to_local() returns shape [1024, 4096]
                                  # each rank holds rows 0:1024, 1024:2048, etc.
```

The logical shape of the parameter is preserved. The DTensor knows that it is `Shard(dim=0)` across 4 ranks. Tools that call `param.shape` see `[4096, 4096]`, which makes compatibility with existing code far easier.

**How FSDP2 hooks work:** Instead of wrapping the module in a new class, `fully_shard` registers pre-forward and post-forward hooks on the original module class using PyTorch's `contract` decorator. The module's `type` is "unioned" in-place: if `model` is a `nn.Linear`, after `fully_shard` it becomes `FSDPLinear`, which is simultaneously an `nn.Linear` and an `FSDPModule`. All existing `nn.Linear` methods still work. The `FSDPModule` mixin adds FSDP2-specific APIs like `reshard()` and `unshard()`.

```
Before fully_shard:
  type(layer.self_attn.q_proj) == nn.Linear

After fully_shard:
  type(layer.self_attn.q_proj) == FSDPLinear  # which is both Linear and FSDPModule
  isinstance(layer.self_attn.q_proj, nn.Linear)  # True
  isinstance(layer.self_attn.q_proj, FSDPModule)  # True
```

**The all-gather/reshard cycle in FSDP2:** The pre-forward hook on `FSDPModule` calls `module.unshard()`, which runs an all-gather, converting the `DTensor` local shard to a full `torch.Tensor`. The post-forward hook calls `module.reshard()`, which frees the full tensor and restores the `DTensor` shard. This is mechanically similar to FSDP1, but the implementation is cleaner: no `record_stream`, no `untyped_storage().resize_(0)` hack.

**FSDP2's memory management:** Instead of `record_stream`, FSDP2 uses explicit stream-to-stream synchronization. After an all-gather completes in the FSDP communication stream, FSDP2 inserts a `cudaStreamWaitEvent` in the compute stream before the module forward runs. This ensures the compute stream waits for the communication to finish without telling the memory allocator anything non-deterministic. The result is deterministic peak memory usage.

### Side-by-Side Comparison

| Property | FSDP1 | FSDP2 |
|---|---|---|
| Sharding primitive | `FlatParameter` (1D concat) | `DTensor` (per-parameter) |
| Module wrapping | New `FullyShardedDataParallel` class | In-place type union via `contract` |
| Parameter names in state dict | Requires `use_orig_params=True` | Always clean, no prefix added |
| Mixed dtype support | Hacks required | Native (each DTensor has its own dtype) |
| Frozen parameter efficiency | All-gathers frozen params too | Can skip communication for frozen params |
| LoRA compatibility | Fragile | Works out of the box |
| Memory management | `record_stream` (non-deterministic) | Stream-to-stream sync (deterministic) |
| Sharded state dict | Requires extra all-gather | Communication-free, DTensor-based |
| API style | Constructor wrapping | Functional, in-place mutation |

---

## 11. FSDP2 in Accelerate: Source-Level Details

**File:** `src/accelerate/utils/dataclasses.py`, `src/accelerate/accelerator.py`

### The fsdp_version Field

The `fsdp_version` field in `FullyShardedDataParallelPlugin` is the toggle. When it is `2`, Accelerate switches its entire FSDP code path:

1. `prepare_model()` calls `_prepare_fsdp2_model()` instead of constructing `FullyShardedDataParallel`.
2. `save_state()` uses `torch.distributed.checkpoint` (DCP) instead of the FSDP1 state dict context managers.
3. The `reshard_after_forward` field is interpreted as a `bool` instead of a `ShardingStrategy` enum.
4. Several FSDP1-only options (`backward_prefetch`, `sync_module_states`, `use_orig_params`) are silently ignored or raise warnings.

### The DeviceMesh

FSDP2 uses `DeviceMesh` for rank group management. By default, Accelerate creates a 1D mesh:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (accelerator.num_processes,))
# This creates a 1D mesh where each rank corresponds to one GPU
# For 4 GPUs: mesh = [[0, 1, 2, 3]]
```

For hybrid sharding (intra-node FSDP, inter-node replication), a 2D mesh is used:

```python
# 2 nodes, 4 GPUs each = 8 total GPUs
# Dim 0: inter-node (replication), Dim 1: intra-node (sharding)
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))
```

The `ParallelismConfig` and `dp_shard_size` parameter in recent Accelerate versions allow controlling the mesh without directly constructing it.

### The bottom-up Wrapping Order

A critical correctness requirement in FSDP2: `fully_shard` must be applied to sub-modules **before** the parent. If you call `fully_shard(root_model)` first and then `fully_shard(root_model.layers[0])`, the sub-module sharding is ignored because the parent's hooks already run first.

Accelerate's `_prepare_fsdp2_model` respects this by iterating sub-modules in a bottom-up order before applying `fully_shard` to the root:

```python
# Apply wrapping bottom-up
for name, module in model.named_modules():
    if should_wrap(module):
        fully_shard(module, mesh=mesh, reshard_after_forward=plugin.reshard_after_forward)

# Root last
fully_shard(model, mesh=mesh, reshard_after_forward=plugin.reshard_after_forward)
```

### The Model+Optimizer Constraint

When `fsdp_version=2`, the `prepare()` method requires that the model and optimizer be passed in the same call:

```python
# This raises ValueError with FSDP2:
model = accelerator.prepare(model)
optimizer = accelerator.prepare(optimizer)  # ERROR: DTensor params already bound

# This is the correct FSDP2 pattern:
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)
```

Internally, Accelerate detects if only a model (no optimizer) is passed with `fsdp_version=2` and raises:

```
ValueError: When using FSDP2, a model and optimizer must be passed together
to Accelerator.prepare() as the optimizer needs to have its parameters
modified after the model is converted.
```

This is a real pain point surfaced by several GitHub issues (e.g., `huggingface/transformers#39961`), where users call `trainer.evaluate()` before `trainer.train()` and trigger the prepare constraint because `evaluate` tries to prepare the model alone.

### Prefetching in FSDP2

FSDP2 implements implicit prefetching by default: the CPU thread issues the all-gather for layer `i+1` while the GPU is computing layer `i`. This is automatic and does not require any configuration, unlike FSDP1's `backward_prefetch` knob.

For advanced users, FSDP2 exposes explicit prefetching via `FSDPModule` APIs:

```python
for i, layer in enumerate(model.layers):
    # Manually prefetch next layer before current layer computes
    if i + 1 < len(model.layers):
        model.layers[i + 1].unshard(async_op=True)  # non-blocking all-gather

    output = layer(input)
    input = output
```

The `set_requires_gradient_sync` API replaces FSDP1's `no_sync()` context manager for gradient accumulation (discussed in Section 16).

---

## 12. FSDP2 Mixed Precision and FP8

FSDP2's mixed precision is specified via `MixedPrecisionPolicy` (not `MixedPrecision` as in FSDP1):

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,   # parameters stored and communicated in bf16
    reduce_dtype=torch.float32,   # gradients reduced in fp32 for stability
)

plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,
    mixed_precision_policy=mp_policy,
)
```

`MixedPrecisionPolicy` supports an additional `output_dtype` field (sets the dtype of the module's output tensor) and `cast_forward_inputs` (whether to cast input tensors to `param_dtype` before forward). These are more granular than FSDP1's `MixedPrecision`.

### FP8 All-Gather (FSDP2 + TorchAO)

A significant FSDP2 feature unique to Accelerate is the `enable_fsdp_float8_all_gather` option. When enabled with TorchAO installed, parameters are cast to FP8 (8-bit floating point) before the all-gather:

```python
plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,
    enable_fsdp_float8_all_gather=True,
    # Optionally customize which layers are FP8:
    # filter_layer_for_fsdp_float8_all_gather=my_filter_fn,
)
```

The bandwidth saving is 50% compared to BF16 all-gather: FP8 uses 1 byte per value vs BF16's 2 bytes. This matters most on inter-GPU links where bandwidth is the bottleneck (PCIe-connected GPUs, or across nodes). On NVLink-connected GPUs where bandwidth exceeds compute, the benefit is smaller.

The `filter_linear_layers` function from `accelerate.utils.ao` is the default filter; it applies FP8 all-gather only to `nn.Linear` layers, leaving embedding and normalization layers in BF16. Embedding tables have integer-valued indices and benefit from precise weight storage, while normalization layer parameters are small enough that the communication saving is minimal.

---

## 13. FSDP2 Checkpointing with DTensor

FSDP2's sharded checkpointing uses `torch.distributed.checkpoint` (DCP) exclusively. The DTensor-based state dict is communication-free: each rank can extract its own shard without coordinating with other ranks.

### Saving with DCP

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

# Get sharded state dict (no communication required)
model_state, optim_state = get_state_dict(model, optimizer)

state_dict = {
    "model": model_state,
    "optimizer": optim_state,
    "step": global_step,
}

dcp.save(state_dict, checkpoint_id="./checkpoint_dir/")
```

The resulting directory contains `.distcp` files, one per rank:

```
checkpoint_dir/
  __0_0.distcp   # rank 0's model shard
  __1_0.distcp   # rank 1's model shard
  __0_0.distcp   # rank 0's optimizer shard
  .metadata      # structural metadata for resharding
```

### Loading with DCP

```python
from torch.distributed.checkpoint.state_dict import set_state_dict

# Initialize fresh model and optimizer (sharded but untrained)
model, optimizer = accelerator.prepare(model, optimizer)

state_dict = {"model": {}, "optimizer": {}}
dcp.load(state_dict, checkpoint_id="./checkpoint_dir/")

set_state_dict(
    model,
    optimizer,
    model_state_dict=state_dict["model"],
    optim_state_dict=state_dict["optimizer"],
)
```

DCP load automatically handles resharding: if you saved with 8 GPUs and load with 4, DCP reads the `.metadata` file to understand the original shard layout and distributes parameters correctly across the new rank count. This is a significant advantage over FSDP1's sharded state dict, which required the same number of ranks for save and load.

### Asynchronous Checkpointing

FSDP2 supports asynchronous checkpointing, where parameters are first copied to pinned CPU memory, after which the main training thread continues while a background thread writes to disk:

```python
import torch.distributed.checkpoint as dcp

# Async save: returns a Future, training continues immediately
future = dcp.async_save(state_dict, checkpoint_id="./checkpoint_dir/")

# Later, wait for the save to complete if needed
future.result()
```

This virtually eliminates checkpoint overhead from the critical training path for models that fit in CPU RAM.

---

## 14. CPU RAM-Efficient Loading

One of the trickiest problems with FSDP is loading a large model's weights at startup. The naive approach creates an OOM on every GPU:

```python
# WRONG for large models with FSDP
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-70b")
# ^ creates full model on every GPU simultaneously: N × 70B × 2 bytes of GPU memory
model = accelerator.prepare(model)
```

`cpu_ram_efficient_loading=True` in the plugin instructs Accelerate to load the model only on rank 0, synchronize weights to other ranks via `sync_module_states`, and shard immediately:

```yaml
fsdp_cpu_ram_efficient_loading: true
fsdp_sync_module_states: true   # FSDP1 only; FSDP2 does this automatically
```

In Python:

```python
plugin = FullyShardedDataParallelPlugin(
    cpu_ram_efficient_loading=True,
)
accelerator = Accelerator(fsdp_plugin=plugin)

# Load on rank 0 only, meta device on other ranks
with accelerator.main_process_first():
    if accelerator.is_main_process:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True
        )
    else:
        # Initialize on meta device: no actual memory allocated
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

# prepare() sees that sync_module_states=True and broadcasts
# rank 0's parameters to all other ranks before sharding
model = accelerator.prepare(model)
```

Without `cpu_ram_efficient_loading`, the 70B model in bf16 requires 140 GB of CPU RAM per GPU just for loading, before FSDP even starts. With `cpu_ram_efficient_loading`, only rank 0 pays the CPU RAM cost, and other ranks pay only the per-shard GPU memory cost.

---

## 15. Activation Checkpointing with FSDP

Activation checkpointing (also called gradient checkpointing) trades compute for memory by not saving activations during the forward pass and recomputing them during the backward pass. Combined with FSDP, it provides another orthogonal dimension of memory savings.

Accelerate enables activation checkpointing via the plugin:

```yaml
fsdp_activation_checkpointing: true
```

Or in Python:

```python
plugin = FullyShardedDataParallelPlugin(
    activation_checkpointing=True,
)
```

When this is set, Accelerate calls `torch.distributed.algorithms.join.Join.enable_hook` and uses `checkpoint_wrapper` from `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` to wrap each FSDP unit. The wrapped module's forward computation is run inside `torch.utils.checkpoint.checkpoint`, which discards activations and recomputes them during backward.

The interaction between FSDP and activation checkpointing is non-trivial at the implementation level: during the recompute phase of the backward pass, FSDP's pre-backward hooks must run again to all-gather parameters. FSDP handles this correctly by registering hooks at the `autograd.Function` level, not just at the module level, so the recompute triggers a fresh all-gather.

### Memory Math with Both

Starting from the FSDP savings:

```
Without activation checkpointing:
  activations = batch_size × seq_len × hidden_size × num_layers × dtype_bytes

With activation checkpointing:
  activations ≈ batch_size × seq_len × hidden_size × dtype_bytes
  # Only one layer's activations live at a time (during recompute)
```

The combination of FSDP (shards parameters) and activation checkpointing (discards activations) allows training models that are 10x-20x larger than would fit naively on the same hardware, at a cost of roughly 30-40% additional compute per step.

---

## 16. Gradient Accumulation with FSDP

Gradient accumulation in FSDP requires special handling to avoid unnecessary communication. In DDP, `no_sync()` prevents the all-reduce that happens at the end of each backward pass when you accumulate gradients. In FSDP, the equivalent mechanism is different for FSDP1 and FSDP2.

### FSDP1: Using no_sync()

```python
accumulation_steps = 4

for step, batch in enumerate(dataloader):
    if step % accumulation_steps == 0:
        context = contextlib.nullcontext()   # last accumulation step: sync
    else:
        context = model.no_sync()            # intermediate steps: skip reduce-scatter

    with context:
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        accelerator.backward(loss)

    if step % accumulation_steps == accumulation_steps - 1:
        optimizer.step()
        optimizer.zero_grad()
```

`model.no_sync()` in FSDP1 suppresses the reduce-scatter that normally happens after each backward. Without it, you would reduce-scatter gradients after every micro-batch and then accumulate the result, which is both wasteful (unnecessary communication) and incorrect in some FSDP configurations.

With Accelerate's `gradient_accumulation_steps` argument, this is handled automatically:

```python
accelerator = Accelerator(gradient_accumulation_steps=4)
```

Accelerate's `GradientState` tracks the accumulation step and switches the context manager automatically.

### FSDP2: set_requires_gradient_sync

FSDP2 replaces `no_sync()` with a more explicit API:

```python
# Disable gradient synchronization for intermediate accumulation steps
for i, module in enumerate(model.modules()):
    if isinstance(module, FSDPModule):
        module.set_requires_gradient_sync(False)

# Run forward + backward (no reduce-scatter happens)
outputs = model(**batch)
loss = outputs.loss / accumulation_steps
loss.backward()

# Re-enable and trigger the final synchronized backward
for module in model.modules():
    if isinstance(module, FSDPModule):
        module.set_requires_gradient_sync(True)

outputs = model(**batch)
loss = outputs.loss / accumulation_steps
loss.backward()  # reduce-scatter now happens
optimizer.step()
```

Accelerate abstracts this with the same `gradient_accumulation_steps` argument, and internally uses `set_requires_gradient_sync` when FSDP2 is detected.

---

## 17. Complete Training Example

This section presents a complete, runnable FSDP1 training script written in the style Accelerate expects. It uses BERT fine-tuning on GLUE MRPC (the same task used in the official `fsdp_with_peak_mem_tracking.py` example).

```python
"""
fsdp_training_example.py

FSDP1 fine-tuning of BERT-large on GLUE MRPC.
Run with:
    accelerate launch --config_file fsdp_config.yaml fsdp_training_example.py
"""

import functools
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertLayer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
import torch

# --- Config ---
MODEL_NAME = "bert-large-uncased"
TASK = "mrpc"
BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 3
LR = 2e-5

# --- FSDP Plugin Setup ---
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Wrap each BertLayer (transformer block) as its own FSDP unit.
# The embedding and pooler layers land in the outermost FSDP root unit.
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={BertLayer},
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision_policy=mixed_precision_policy,
    state_dict_type="SHARDED_STATE_DICT",   # recommended for checkpointing
    cpu_ram_efficient_loading=True,
)

# --- Accelerator ---
accelerator = Accelerator(
    fsdp_plugin=fsdp_plugin,
    gradient_accumulation_steps=GRAD_ACCUM,
    log_with="tensorboard",
    project_dir="./logs",
)
accelerator.init_trackers("fsdp_bert_mrpc")

# --- Tokenizer and Dataset ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_datasets = load_dataset("glue", TASK)

def tokenize_fn(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        max_length=128,
        truncation=True,
    )

tokenized = raw_datasets.map(tokenize_fn, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataloader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE)

# --- Model ---
# Use low_cpu_mem_usage=True with cpu_ram_efficient_loading
with accelerator.main_process_first():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        low_cpu_mem_usage=True,
    )

# --- Optimizer and Scheduler ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

num_training_steps = (len(train_dataloader) // GRAD_ACCUM) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps,
)

# --- prepare() ---
# All objects must be prepared together.
# For FSDP, prepare() wraps the model in FullyShardedDataParallel,
# replaces the DataLoader sampler with DistributedSampler,
# and wraps the scheduler with AcceleratedScheduler.
model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    peak_mem_before = torch.cuda.max_memory_allocated()

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.detach().float()

    peak_mem_after = torch.cuda.max_memory_allocated()
    peak_delta_gb = (peak_mem_after - peak_mem_before) / 1e9

    accelerator.print(
        f"Epoch {epoch}: loss={total_loss / len(train_dataloader):.4f}, "
        f"peak memory this epoch = {peak_delta_gb:.2f} GB"
    )

    # --- Evaluation ---
    model.eval()
    correct = 0
    total = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        # gather_for_metrics collects predictions from all ranks
        predictions, labels = accelerator.gather_for_metrics(
            (predictions, batch["labels"])
        )
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    accuracy = correct / total
    accelerator.print(f"Epoch {epoch}: eval accuracy = {accuracy:.4f}")
    accelerator.log({"eval_accuracy": accuracy, "epoch": epoch})

    # --- Checkpoint ---
    accelerator.save_state(f"./checkpoints/epoch_{epoch}/")

accelerator.end_training()
```

The accompanying YAML config (`fsdp_config.yaml`):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
mixed_precision: bf16
num_processes: 2
```

---

## 18. Complete FSDP2 Training Example

```python
"""
fsdp2_training_example.py

FSDP2 fine-tuning of BERT-large on GLUE MRPC.
Demonstrates the key API differences from FSDP1:
  - fsdp_version=2
  - reshard_after_forward (bool) instead of ShardingStrategy
  - model and optimizer prepared together
  - MixedPrecisionPolicy instead of MixedPrecision
  - SHARDED_STATE_DICT with DCP

Run with:
    accelerate launch --config_file fsdp2_config.yaml fsdp2_training_example.py
"""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin

MODEL_NAME = "bert-large-uncased"
TASK = "mrpc"
BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 3
LR = 2e-5

# --- FSDP2 Plugin ---
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # fp32 reduction for numerical stability
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,                         # <-- key switch
    reshard_after_forward=True,             # bool in FSDP2 (True = FULL_SHARD)
    mixed_precision_policy=mp_policy,
    cpu_ram_efficient_loading=True,
    state_dict_type="SHARDED_STATE_DICT",
    auto_wrap_policy="TRANSFORMER_BASED_WRAP",
    # fsdp_transformer_layer_cls_to_wrap is read from config YAML
)

accelerator = Accelerator(
    fsdp_plugin=fsdp_plugin,
    gradient_accumulation_steps=GRAD_ACCUM,
)

# --- Dataset ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_datasets = load_dataset("glue", TASK)

def tokenize_fn(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"],
        padding="max_length", max_length=128, truncation=True,
    )

tokenized = raw_datasets.map(tokenize_fn, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataloader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE)

# --- Model ---
with accelerator.main_process_first():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, low_cpu_mem_usage=True,
    )

# --- Optimizer must be created BEFORE prepare() ---
# After prepare(), parameters become DTensors. We create the optimizer
# on the original parameters, then pass both to prepare() which
# re-initializes the optimizer's param_groups to point at DTensors.
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# --- prepare(): model and optimizer MUST be together for FSDP2 ---
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Scheduler is created after prepare() because it uses optimizer
num_steps = (len(train_dataloader) // GRAD_ACCUM) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_steps // 10, num_steps)
scheduler = accelerator.prepare(scheduler)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch in train_dataloader:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.detach().float()

    accelerator.print(f"Epoch {epoch}: loss={running_loss / len(train_dataloader):.4f}")

    # --- Checkpoint (FSDP2: DCP-based, no communication required) ---
    accelerator.save_state(f"./checkpoints/fsdp2_epoch_{epoch}/")

    # --- Eval ---
    model.eval()
    correct = total = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        preds, labels = accelerator.gather_for_metrics((preds, batch["labels"]))
        correct += (preds == labels).sum().item()
        total += labels.numel()

    accelerator.print(f"Epoch {epoch}: eval accuracy={correct / total:.4f}")

accelerator.end_training()
```

YAML config for FSDP2 (`fsdp2_config.yaml`):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_reshard_after_forward: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_activation_checkpointing: false
mixed_precision: bf16
num_processes: 2
```

To convert an existing FSDP1 config to FSDP2:

```bash
accelerate to-fsdp2 --config_file fsdp_config.yaml --output_file fsdp2_config.yaml
```

---

## 19. Contribution Opportunities

Each of these is a concrete, scoped area where the Accelerate codebase has open work or known gaps as of early 2026.

### FSDP2 Edge Cases in Checkpoint Conversion

The `accelerate to-fsdp2` CLI tool converts FSDP1 YAML configs to FSDP2 equivalents. However, it only handles the options in the known equivalence table. Any custom `mixed_precision_policy` objects created in Python code are not converted by the tool. A contribution could add conversion support for `MixedPrecision` -> `MixedPrecisionPolicy` translation in the CLI tool, plus a warning when fields are non-trivially different (e.g., `buffer_dtype` has no direct equivalent in `MixedPrecisionPolicy`).

**Files to look at:** `src/accelerate/commands/config/config_args.py`, `src/accelerate/utils/dataclasses.py`

### FSDP2 + Activation Checkpointing Interaction

`fsdp_activation_checkpointing: true` works in FSDP1 via `checkpoint_wrapper`. The FSDP2 path does not use `checkpoint_wrapper` but instead relies on PyTorch's `torch.utils.checkpoint.checkpoint` applied to each `FSDPModule`. The integration is incomplete for certain gradient accumulation configurations: when `set_requires_gradient_sync(False)` is active and activation recompute is triggered, there are cases where the pre-backward all-gather is skipped. This is a genuine correctness bug worth investigating.

**Files to look at:** `src/accelerate/utils/fsdp_utils.py`, `src/accelerate/accelerator.py`, `tests/fsdp/test_fsdp.py`

### Benchmarking FSDP Strategy Comparison

There is no official Accelerate benchmark that systematically compares `FULL_SHARD` vs `SHARD_GRAD_OP` vs `HYBRID_SHARD` across model sizes and GPU counts. A contribution could add a `benchmarks/fsdp_strategy_comparison.py` script that measures peak memory and step time for each strategy on a standardized model configuration, enabling contributors and users to make informed strategy decisions.

### FSDP2 `ignored_params` for LoRA

In FSDP1, the `ignored_modules` parameter excludes specific modules from sharding, which is critical for QLoRA (where the base model is sharded but the LoRA adapters should not be). FSDP2 has `ignored_params` at the parameter level (not module level), which is more granular. Accelerate's `FullyShardedDataParallelPlugin` exposes `ignored_modules` for FSDP1 but does not yet expose `ignored_params` for FSDP2. Adding this, with appropriate tests, is a well-scoped feature contribution.

**Files to look at:** `src/accelerate/utils/dataclasses.py`, `tests/fsdp/test_fsdp.py`

### Documentation: FSDP2 Optimizer Constraint

The `ValueError` raised when model and optimizer are prepared separately under FSDP2 (discussed in section 11) is cryptic to new users who don't understand why the constraint exists. A contribution to the error message to explain the DTensor parameter initialization requirement, plus a note in the FSDP documentation page, would prevent a common support question. The `huggingface/transformers#39961` issue shows this hits real users repeatedly.

---

## 20. What You Should Know After This Chapter

Work through these questions without looking anything up. If you cannot answer one, re-read the corresponding section.

**Q: What is the memory formula for FSDP FULL_SHARD on N GPUs for a model with P parameters in fp32 with Adam?**

A: Sharded state: `(P × 4 bytes_params + P × 4 bytes_grads + P × 8 bytes_adam) / N`. Plus a transient working set equal to the memory of the largest FSDP unit during the all-gather. The sharded per-GPU footprint is approximately `16P_bytes / N`, where `16` comes from `4 + 4 + 8`.

**Q: What is the difference between all-gather and reduce-scatter? When does FSDP use each?**

A: All-gather: every rank contributes its shard, every rank receives the full concatenated tensor. FSDP uses this before each layer's forward (and before backward in FULL_SHARD). Reduce-scatter: reduces (sums) a full tensor across all ranks, then distributes shards (each rank receives only its fraction of the sum). FSDP uses this after backward to synchronize and re-shard gradients.

**Q: What is `FlatParameter` and why did FSDP2 eliminate it?**

A: In FSDP1, all parameters of a wrapped module are concatenated into one 1D `FlatParameter` before sharding. This loses per-parameter metadata (dtype, requires_grad), makes partial freezing (LoRA) require extra workarounds, produces confusing state dict keys, and uses `record_stream` for memory management which is non-deterministic. FSDP2 uses `DTensor` instead, sharding each parameter individually while preserving its shape, dtype, and name.

**Q: Why must the optimizer be passed to `prepare()` at the same time as the model when using FSDP2?**

A: `fully_shard()` converts model parameters from `torch.Tensor` to `DTensor` in-place. An optimizer created before `fully_shard()` holds references to the original `torch.Tensor` objects. After sharding, those tensors no longer exist as model parameters. The optimizer must be re-initialized (or its `param_groups` re-bound) to the `DTensor` shards after `fully_shard()` completes. Accelerate does this re-binding internally when both are passed together.

**Q: What is the difference between `FULL_STATE_DICT` and `SHARDED_STATE_DICT`? When should you use each?**

A: `FULL_STATE_DICT` triggers an all-gather to rank 0, producing a single checkpoint file loadable without FSDP. It requires enough CPU RAM to hold the full model and is slow for large models. Use it for final model export or when you need a checkpoint loadable without distributed setup. `SHARDED_STATE_DICT` has each rank save only its shard via DCP. It is fast, requires no inter-rank communication, and scales to any model size. Use it for intermediate training checkpoints. The tradeoff is that sharded checkpoints require FSDP (or DCP load) to read back.

**Q: What does the `auto_wrap_policy` do and why can you not shard mid-attention?**

A: The `auto_wrap_policy` determines which `nn.Module` instances become FSDP units (sharding boundaries). The backward pass of attention is coupled across Q, K, V, and O projections: the gradient of Q requires O's accumulated gradient, so all four must be in memory simultaneously during backward. Wrapping them in separate FSDP units would require all four to be all-gathered independently while still keeping them all active, which creates a correctness problem. The safe boundary is the full transformer block.

**Q: What is `HYBRID_SHARD` and when is it the right choice?**

A: HYBRID_SHARD shards parameters across GPUs within a node (using fast NVLink) but keeps a full replica on each node (replicating across the slow inter-node link). The cross-node communication is a DDP-style all-reduce, not an all-gather, which is more efficient per byte on typical InfiniBand or Ethernet interconnects. Use it in multi-node training when inter-node bandwidth is the bottleneck, which it almost always is in standard cloud configurations.

**Q: What is the order requirement for applying `fully_shard()` in FSDP2?**

A: Sub-modules must be sharded before their parent. Calling `fully_shard(root)` before `fully_shard(root.layers[0])` means `root`'s hooks fire first, and when `root.layers[0].forward` is called, no FSDP hooks are registered on it because the sub-module was never sharded. The correct pattern is to iterate sub-modules in depth-first order (bottom-up), then shard the root last.

---

## Source Reading Guide

Below is a prioritized reading order for the files in this chapter's scope, with specific line number ranges as of early 2026.

**1. `src/accelerate/utils/dataclasses.py`**
Search for `class FullyShardedDataParallelPlugin`. Read the dataclass fields (all `fsdp_*` attributes), then `__post_init__`, then `set_auto_wrap_policy`. This is the central configuration object.

**2. `src/accelerate/accelerator.py`**
Search for `prepare_model`. Read the entire method, focusing on the `DistributedType.FSDP` branch. Then find `_prepare_fsdp1_model` and `_prepare_fsdp2_model` (or their equivalents if the method was inlined). Trace the kwargs being assembled for `FullyShardedDataParallel(model, **kwargs)` in FSDP1 and `fully_shard(module, **kwargs)` in FSDP2.

**3. `src/accelerate/utils/fsdp_utils.py`**
Read `load_fsdp_model`, `save_fsdp_model`, and `merge_fsdp_weights`. These are the checkpointing helpers that handle the StateDictType context managers. Pay attention to the `rank0_only` and `offload_to_cpu` arguments.

**4. `examples/by_feature/fsdp_with_peak_mem_tracking.py`**
Read the complete example. Note where `torch.cuda.max_memory_allocated()` and `torch.cuda.reset_peak_memory_stats()` are called. This is the official Accelerate FSDP example and the style you should follow for new examples.

**5. `tests/fsdp/test_fsdp.py`**
Read the `test_fsdp_*` test functions. Notice how they use `launch_command_kwargs` to set sharding strategy, state dict type, and wrapping policy from outside the script. Look for any tests that use `cpu_ram_efficient_loading` and `sync_module_states`. These tests are multi-process tests and use `accelerate.test_utils.testing.run_first` or similar harnesses.

**6. `src/accelerate/commands/config/config_args.py`**
Search for `to_fsdp2`. This is the CLI command implementation that converts FSDP1 configs to FSDP2. Reading this will show you exactly which field translations are implemented and which are missing.
