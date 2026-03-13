# Accelerate Internals: Level 4 — Mixed Precision Training

> **Series:** Contributing to HuggingFace Accelerate — A Developer's Field Guide
> **Chapter:** 4 of N — Mixed Precision Training
> **Source references:** `src/accelerate/accelerator.py`, `src/accelerate/utils/dataclasses.py`,
> `src/accelerate/utils/transformer_engine.py`, `src/accelerate/utils/ao.py`

---

## Table of Contents

1. [Why Precision Matters](#1-why-precision-matters)
2. [The Three Floating-Point Formats](#2-the-three-floating-point-formats)
3. [The Accelerate Interface](#3-the-accelerate-interface)
4. [How Accelerate Initializes Mixed Precision](#4-how-accelerate-initializes-mixed-precision)
5. [The `backward()` Method: Three Code Paths](#5-the-backward-method-three-code-paths)
6. [GradScaler Deep Dive (fp16)](#6-gradscaler-deep-dive-fp16)
7. [The bf16 Path: No Scaler, but `native_amp`](#7-the-bf16-path-no-scaler-but-native_amp)
8. [Autocast: What It Actually Does](#8-autocast-what-it-actually-does)
9. [The `AutocastKwargs` Handler](#9-the-autocastkwargs-handler)
10. [FP8 Training Architecture](#10-fp8-training-architecture)
11. [TransformerEngine Backend (`TERecipeKwargs`)](#11-transformerengine-backend-terecipekwargs)
12. [TorchAO Backend (`AORecipeKwargs`)](#12-torchao-backend-aorecipekwargs)
13. [MS-AMP Backend (`MSAMPRecipeKwargs`) — Deprecated](#13-ms-amp-backend-msampreipekwargs--deprecated)
14. [How Accelerate Wires FP8 Into `prepare_model()`](#14-how-accelerate-wires-fp8-into-prepare_model)
15. [Checkpointing the Scaler](#15-checkpointing-the-scaler)
16. [Hardware Compatibility Matrix](#16-hardware-compatibility-matrix)
17. [Practical Exercises](#17-practical-exercises)
18. [What You Should Understand After This Level](#18-what-you-should-understand-after-this-level)
19. [Source Reading Guide](#19-source-reading-guide)

---

## 1. Why Precision Matters

Every floating-point number on a GPU is stored in a fixed number of bits. IEEE 754 fp32, the historical default for deep learning, uses 32 bits: 1 sign bit, 8 exponent bits, and 23 mantissa bits. The mantissa controls *precision* (how exactly you can represent a value) while the exponent controls *dynamic range* (the span from the smallest representable nonzero value to the largest).

Training with fp32 is safe but wasteful. Modern GPU tensor cores are optimized for half-precision arithmetic and can run fp16 or bf16 matrix multiplications 2-8x faster than fp32 while consuming half the memory per parameter and activation. The catch is that reduced precision introduces risk: very small gradients can be rounded to zero (underflow) or very large values can be clamped to infinity (overflow), corrupting training.

Mixed precision training resolves the trade-off by doing most arithmetic in a reduced format while keeping weights and gradient accumulation in fp32. The key design question is *where* precision reduction is safe. For fp16 the answer involves dynamic range management (gradient scaling). For bf16 the answer is structural -- bf16 has the same exponent width as fp32, so the dynamic range problem simply does not arise.

The word "mixed" in mixed precision is critical. At no point does Accelerate train an entirely fp16 model. The canonical approach (originally described in the NVIDIA paper "Mixed Precision Training," Micikevicius et al. 2018) is:

- Forward pass: compute activations in fp16/bf16
- Backward pass: compute gradients in fp16/bf16
- Weight storage: keep fp32 master copies
- Weight update: apply fp32 gradients to fp32 masters, then cast the updated masters back to fp16/bf16 for the next forward pass

PyTorch autocast handles casting automatically for eligible ops. The fp32 master copy is maintained by the optimizer's param groups. Everything stays consistent because `AcceleratedOptimizer` (reviewed in Chapter 1) defers the step until `sync_gradients` is True and, for fp16, the scaler signals that no inf/nan was detected.

---

## 2. The Three Floating-Point Formats

Before reading source code, you need a concrete mental model of what each format is and why it behaves differently.

### fp32 (baseline)

```
 1 sign | 8 exponent | 23 mantissa = 32 bits
 dynamic range: ~1.2e-38 to ~3.4e+38
 smallest positive normal: 1.2e-38
 precision: ~7 decimal digits
```

fp32 is the reference. Everything works. Nothing overflows or underflows during normal training.

### fp16

```
 1 sign | 5 exponent | 10 mantissa = 16 bits
 dynamic range: ~6e-8 to ~65504
 smallest positive normal: ~6.1e-5
 precision: ~3.3 decimal digits
```

The key number is **65504**. That is the maximum representable fp16 value. Gradients of typical neural networks can easily exceed this, causing overflow. More commonly, small gradients -- particularly in later epochs when the network has largely converged -- fall below `6e-5` and flush to zero (underflow). GradScaler compensates for both by artificially inflating the loss before backward and deflating gradients before the optimizer step.

The 5-exponent field is the root cause. With only 5 bits for the exponent, the range from smallest to largest is compressed to roughly 4 orders of magnitude compared to fp32's 76 orders of magnitude.

### bf16 (bfloat16)

```
 1 sign | 8 exponent | 7 mantissa = 16 bits
 dynamic range: ~1.2e-38 to ~3.4e+38  (same as fp32)
 smallest positive normal: ~1.2e-38
 precision: ~2.3 decimal digits
```

Google Brain designed bf16 by taking fp32's exponent field exactly as-is and truncating the mantissa to 7 bits. The result is that bf16 can represent the same range of magnitudes as fp32 -- gradients that fit in fp32 always fit in bf16. The cost is precision: only 7 mantissa bits means each value is rounded to about 2.3 decimal digits. This is sufficient for gradient updates, which are inherently noisy, but causes more rounding per operation than fp16 would.

Because bf16 has the same exponent, **no gradient scaling is needed**. This makes bf16 dramatically simpler to use and less fragile. It also means bf16 training is essentially immune to the gradient overflow/underflow instability that fp16 users must manage with GradScaler.

The catch: bf16 requires Ampere or later NVIDIA GPUs (A100, A10, RTX 3090, etc.). Pascal (P100) and Volta (V100) do not support bf16 natively, though they can emulate it in software at a significant performance cost.

### fp8 (E4M3 and E5M2)

```
 E4M3:  1 sign | 4 exponent | 3 mantissa = 8 bits
        max value: 448
        used for: forward pass activations and weights

 E5M2:  1 sign | 5 exponent | 2 mantissa = 8 bits
        max value: 57344
        used for: backward pass gradients
```

FP8 cuts the width in half again. With only 3-4 mantissa bits the representable range is tiny. Scaling is mandatory and the scaling strategy is different from fp16 -- instead of one global loss scale, FP8 uses per-tensor scaling factors computed from a history of recent `amax` (absolute maximum) values. This is called *delayed scaling* and it is handled entirely by TransformerEngine or TorchAO, not by PyTorch's native GradScaler.

The HYBRID format (used by TransformerEngine's `DelayedScaling`) uses E4M3 for the forward pass (higher precision per value) and E5M2 for the backward pass (higher range, needed because backward gradients span a wider magnitude range than forward activations).

---

## 3. The Accelerate Interface

The public API is intentionally simple:

```python
from accelerate import Accelerator

# Option 1: string shorthand
accelerator = Accelerator(mixed_precision="bf16")

# Option 2: fp16 with custom GradScaler parameters
from accelerate.utils import GradScalerKwargs

scaler_kwargs = GradScalerKwargs(
    init_scale=2**16,        # initial loss multiplier (default 65536)
    growth_factor=2.0,       # how aggressively to increase scale
    backoff_factor=0.5,      # how aggressively to decrease after inf/nan
    growth_interval=2000,    # steps between scale increase attempts
)
accelerator = Accelerator(
    mixed_precision="fp16",
    kwargs_handlers=[scaler_kwargs],
)

# Option 3: fp8 with explicit backend
from accelerate.utils import TERecipeKwargs

fp8_kwargs = TERecipeKwargs(
    backend="te",
    fp8_format="HYBRID",
    amax_history_len=1024,
    amax_compute_algo="max",
)
accelerator = Accelerator(
    mixed_precision="fp8",
    kwargs_handlers=[fp8_kwargs],
)
```

The `mixed_precision` string sets up the entire precision mode. The `kwargs_handlers` list is a heterogeneous bag of configuration objects that Accelerate routes to the right internal handler by class inspection.

All four valid values for `mixed_precision`:

| Value | Meaning |
|-------|---------|
| `"no"` | fp32 everywhere, no autocast |
| `"fp16"` | fp16 forward/backward, fp32 master weights, GradScaler active |
| `"bf16"` | bf16 forward/backward, fp32 master weights, no GradScaler |
| `"fp8"` | depends on backend, requires separate RecipeKwargs |

The value can also be set via the environment variable `ACCELERATE_MIXED_PRECISION`, which is what `accelerate launch --mixed_precision fp16` writes for you.

---

## 4. How Accelerate Initializes Mixed Precision

The initialization logic lives in `Accelerator.__init__()` in `src/accelerate/accelerator.py`. It is structured as a sequence of checks and attribute assignments that wire up the right objects for the selected precision mode.

### Step 1 -- kwargs_handlers dispatch table

Near the top of `__init__`, Accelerate builds a dispatch table mapping handler classes to attribute names:

```python
# src/accelerate/accelerator.py (Accelerator.__init__)

handler_to_attr = {
    GradScalerKwargs:  "scaler_handler",
    InitProcessGroupKwargs: "init_handler",
    FP8RecipeKwargs:   "fp8_recipe_handler",
    AutocastKwargs:    "autocast_handler",
    ProfileKwargs:     "profile_handler",
    AORecipeKwargs:    "ao_recipe_handler",
    TERecipeKwargs:    "te_recipe_handler",
    MSAMPRecipeKwargs: "msamp_recipe_handler",
}
self.has_fp8_handler = False

if kwargs_handlers is not None:
    for handler in kwargs_handlers:
        assert isinstance(handler, KwargsHandler), (
            f"Unsupported kwargs handler passed: {handler}, must be one that "
            f"inherits `accelerate.utils.KwargsHandler`."
        )
        # find which attribute this handler type maps to
        for handler_class, attr_name in handler_to_attr.items():
            if isinstance(handler, handler_class):
                setattr(self, attr_name, handler)
                if "recipe_handler" in attr_name:
                    self.has_fp8_handler = True
                break
```

The handler objects themselves are simple dataclasses inheriting from `KwargsHandler`. The `to_kwargs()` method on each handler converts its fields to a plain dict, which is then unpacked into whatever object the handler is configuring (e.g., `GradScaler(**scaler_handler.to_kwargs())`).

### Step 2 -- fp16 path: create GradScaler

```python
# src/accelerate/accelerator.py (Accelerator.__init__, simplified)

err = (
    "fp16 mixed precision requires {} as it relies on the dedicated hardware "
    "instructions for fast low-precision computations."
)

if self.state.mixed_precision == "fp16":
    if not torch.cuda.is_available():
        raise ValueError(err.format("a GPU"))

    # Pull any custom kwargs from scaler_handler, or use defaults
    kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}

    if self.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        self.scaler = ShardedGradScaler(**kwargs)
    elif is_npu_available():
        self.scaler = torch.npu.amp.GradScaler(**kwargs)
    else:
        self.scaler = torch.cuda.amp.GradScaler(**kwargs)
```

Three things to notice:

1. `self.scaler` is only created for fp16. For bf16 and fp8, it stays `None`.
2. The FSDP path uses `ShardedGradScaler` instead of the standard one. This is necessary because in FSDP the gradient tensors are sharded across ranks, and the standard GradScaler does not know how to inspect sharded gradients for inf/nan. `ShardedGradScaler` performs the inf/nan check across the FSDP shards correctly before deciding whether to skip the optimizer step.
3. The kwargs come from `scaler_handler.to_kwargs()`. If you pass a `GradScalerKwargs` with `init_scale=2**15`, that value lands exactly in `torch.cuda.amp.GradScaler(init_scale=2**15)`.

### Step 3 -- bf16 path: set `native_amp`

```python
elif self.state.mixed_precision == "bf16" and self.distributed_type not in (
    DistributedType.DEEPSPEED,
    DistributedType.MEGATRON_LM,
):
    if self.device.type in ["cpu", "xpu"]:
        self.native_amp = True
    else:
        self.native_amp = is_bf16_available(True)

    if mixed_precision == "bf16" and not self.native_amp and not is_tpu_available():
        raise ValueError(
            err.format("PyTorch >= 1.10 and a supported device.")
        )
```

`self.native_amp = True` is the signal for the bf16 path. Unlike fp16, no scaler is constructed. The flag `native_amp` tells `autocast()` to enable itself and tells `backward()` that no scaling is needed.

Note the guard: DEEPSPEED and MEGATRON_LM skip this block entirely because those backends manage their own precision handling internally.

### Step 4 -- fp8 path: detect and set fp8_backend

```python
# Simplified from accelerator.__init__
if self.mixed_precision == "fp8":
    if self.has_fp8_handler:
        # User specified a backend explicitly via a RecipeKwargs
        if self.te_recipe_handler is not None:
            self.fp8_backend = FP8BackendType.TE
        elif self.ao_recipe_handler is not None:
            self.fp8_backend = FP8BackendType.AO
        elif self.msamp_recipe_handler is not None:
            self.fp8_backend = FP8BackendType.MSAMP
    else:
        # Auto-detect from installed packages
        if is_transformer_engine_available():
            self.fp8_backend = FP8BackendType.TE
        elif is_torchao_available():
            self.fp8_backend = FP8BackendType.AO
        else:
            raise ValueError(
                "Tried to train with `fp8` and auto-detect backend, but no "
                "FP8-compatible backend was installed. "
                "Valid backends are: `torchao`, `transformer-engine`, and `msamp`."
            )

    self.delayed_fp8_autocast = (
        self.fp8_backend == FP8BackendType.TE
        and self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.FSDP)
    )
```

The `delayed_fp8_autocast` flag is worth pausing on. TransformerEngine's fp8 context (`te.fp8_autocast`) wraps the forward pass but it cannot wrap the backward pass separately -- the context must exit before backward begins. In multi-GPU setups, Accelerate therefore delays entering the TE autocast context until the model's forward call begins (by wrapping `model.forward`), rather than wrapping the entire training step. The `delayed_fp8_autocast` flag tells `prepare_model()` to apply this wrapping.

---

## 5. The `backward()` Method: Three Code Paths

`accelerator.backward(loss)` is the single call that replaces `loss.backward()` in user code. Its implementation routes to one of three code paths based on the active precision mode.

```python
# src/accelerate/accelerator.py

def backward(self, loss, **kwargs):
    """
    Scales the gradients in accordance to the GradientAccumulationPlugin and
    calls the correct backward() based on the configuration. Should be used
    in lieu of loss.backward().
    """
    if self.distributed_type != DistributedType.DEEPSPEED:
        # --- Path 1: fp16 ---
        if self.scaler is not None:
            # Scale the loss before backward so gradients are large enough
            # to survive fp16's limited dynamic range
            self.scaler.scale(loss).backward(**kwargs)
        # --- Path 2: bf16 or no mixed precision ---
        else:
            loss.backward(**kwargs)
    else:
        # --- Path 3: DeepSpeed ---
        # DeepSpeed's engine.backward() handles its own scaling internally
        self.deepspeed_engine_wrapped.backward(loss, **kwargs)
```

The simplicity of `backward()` is intentional. The complexity was pushed into the scaler, the autocast context manager, and the model-wrapping in `prepare_model()`. By the time `backward()` is called, the forward pass has already run under autocast (so activations are already in the right dtype), and the only remaining task is to handle gradient scaling on the way out.

The gradient accumulation interaction deserves a note. When `accelerator.accumulate(model)` is active, `backward()` is called on every micro-step but the underlying optimizer step happens only at synchronization boundaries. During non-sync steps, the `scaler.scale(loss).backward()` call accumulates scaled gradients in the parameter `.grad` tensors. At the sync step, `scaler.step(optimizer)` unscales those accumulated gradients before passing them to the optimizer. This means the fp16 gradient scale is consistently applied across accumulation boundaries.

### Annotated Diagram of the fp16 Path

```
Training step N:
                        ┌──────────────┐
   input ──────────────►│ model.forward │ (autocast active: ops run in fp16)
                        └──────┬───────┘
                               │ loss (fp32, computed by loss_fn post-autocast)
                               ▼
                   ┌───────────────────────┐
                   │ scaler.scale(loss)     │  loss ← loss × scale_factor
                   └───────────┬───────────┘        (e.g. ×65536)
                               │ scaled_loss
                               ▼
                   ┌───────────────────────┐
                   │ scaled_loss.backward() │  gradients in fp16, but ×65536
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  scaler.step(optim)    │  internally calls scaler.unscale_()
                   │                        │  gradients ÷ scale_factor → true grads
                   │                        │  checks for inf/nan in every gradient
                   │                        │  if inf/nan found: SKIP optimizer step
                   │                        │  if clean: optimizer.step()
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  scaler.update()       │  adjust scale_factor for step N+1
                   └───────────────────────┘
```

---

## 6. GradScaler Deep Dive (fp16)

`torch.amp.GradScaler` is a PyTorch object but Accelerate integrates it tightly. Understanding its state machine is essential for debugging fp16 instability.

### Constructor defaults

```python
torch.amp.GradScaler(
    device="cuda",
    init_scale=65536.0,    # 2^16 — the initial loss multiplier
    growth_factor=2.0,     # multiply scale by this after growth_interval clean steps
    backoff_factor=0.5,    # multiply scale by this after an inf/nan step
    growth_interval=2000,  # how many consecutive clean steps before growth
    enabled=True,
)
```

These are the defaults. When you pass a `GradScalerKwargs` to Accelerator, its `to_kwargs()` method produces a dict of these fields which is unpacked into the `GradScaler` constructor.

### The inf/nan detection mechanism

During `scaler.step(optimizer)`, GradScaler calls `scaler.unscale_(optimizer)` internally. This divides every parameter gradient by `scale_factor` in-place. After unscaling, GradScaler checks every gradient tensor for inf or nan using a fused CUDA kernel that avoids reading all gradients back to CPU.

The outcome is stored in an internal `_found_inf_per_device` dict -- one entry per CUDA device holding a 0.0 or 1.0 scalar tensor. If any device recorded 1.0, the optimizer step is skipped by replacing the `optimizer.step()` call with a no-op.

```python
# What scaler.step() does internally (conceptual pseudocode)
def step(self, optimizer):
    if not self._enabled:
        return optimizer.step()

    self.unscale_(optimizer)           # divide grads by scale_factor

    # Check every grad for inf/nan
    found_inf = sum(
        v.item() for v in self._found_inf_per_device.values()
    )

    if found_inf:
        # Skip the optimizer step this iteration
        # The optimizer's param .grad tensors have inf/nan,
        # applying them would corrupt the weights
        pass
    else:
        optimizer.step()               # clean gradients, proceed
```

Accelerate exposes whether the optimizer step was skipped via:

```python
accelerator.optimizer_step_was_skipped   # bool, True if inf/nan were found
```

This property is useful for learning rate schedulers: if the step was skipped you probably do not want to advance the scheduler. `AcceleratedScheduler` (from Chapter 1) already respects this internally, only calling `scheduler.step()` when `optimizer_step_was_skipped` is False.

### The scale factor state machine

The scale factor starts at `init_scale` (65536 by default). After that:

- Every `growth_interval` consecutive non-overflow steps: `scale *= growth_factor` (default: double)
- Any step with inf/nan: `scale *= backoff_factor` (default: halve)

The scale factor can fall below 1.0 -- this is counterintuitive but valid. If a model consistently overflows even at scale=1.0, GradScaler will push the scale below 1.0 to keep gradients representable. PyTorch's documentation explicitly states that GradScaler does not guarantee the scale stays above 1.

You can read the current scale at any time:

```python
print(accelerator.scaler.get_scale())    # e.g., 32768.0
```

Over a healthy training run you expect to see:
- Scale starts at 65536
- Drops to 32768 on the first inf/nan encounter
- Climbs back to 65536 after 2000 clean steps
- Oscillates around a stable value once training is warm

A scale that monotonically drops toward 1 or below indicates the model architecture is fundamentally incompatible with fp16 and bf16 should be used instead.

### Customizing GradScaler for your use case

```python
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs

# For a model that frequently overflows early in training:
# Start with a smaller scale and grow more conservatively
scaler_kwargs = GradScalerKwargs(
    init_scale=2**12,          # start small: 4096
    growth_factor=1.5,         # grow slowly
    backoff_factor=0.5,
    growth_interval=500,       # try to grow frequently
)

# For a stable model where you want to maximize scale:
scaler_kwargs = GradScalerKwargs(
    init_scale=2**20,          # 1,048,576 — very large
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=5000,      # grow infrequently, be conservative
)

accelerator = Accelerator(
    mixed_precision="fp16",
    kwargs_handlers=[scaler_kwargs],
)
```

### Gradient clipping with fp16

If you use gradient clipping (`accelerator.clip_grad_norm_()`), the clip must happen *after* unscaling but *before* the optimizer step. Accelerate handles this automatically:

```python
# accelerator.clip_grad_norm_() calls scaler.unscale_() then torch.nn.utils.clip_grad_norm_()
accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If you call `torch.nn.utils.clip_grad_norm_()` directly on scaled gradients (i.e., before `scaler.unscale_()` has been called), you will clip at the wrong magnitude and introduce a systematic training error that is very hard to debug. Always use `accelerator.clip_grad_norm_()`.

---

## 7. The bf16 Path: No Scaler, but `native_amp`

bf16 needs no GradScaler, but it still needs autocast. Autocast is what causes eligible ops (matmul, conv2d, etc.) to execute in bf16 instead of fp32.

The state after init for bf16 is:

```python
self.scaler = None           # no scaler
self.native_amp = True       # flag that tells autocast() to enable itself
```

The `backward()` call for bf16 is simply:

```python
loss.backward(**kwargs)      # plain PyTorch backward, no scaling
```

But the autocast context must still have been active during the forward pass. Accelerate applies autocast via a context manager that wraps the model's forward method -- or alternatively the user wraps their training step with `accelerator.autocast()`. This is the same mechanism for fp16 and bf16, just with different dtype arguments.

### How `accelerator.autocast()` works

```python
# src/accelerate/accelerator.py

@contextmanager
def autocast(self, cache_enabled: bool = True, autocast_handler: AutocastKwargs = None):
    """
    Will apply automatic mixed-precision inside the block, if it is enabled.
    Nothing different will happen otherwise.
    """
    if self.native_amp:
        autocast_context = get_mixed_precision_context_manager(
            self.native_amp, self.autocast_handler
        )
        autocast_context.__enter__()
        yield
        autocast_context.__exit__(None, None, None)
    else:
        yield
```

When `self.native_amp` is False (precision mode is "no"), the context manager is a simple passthrough that does nothing. When `native_amp` is True, it enters the torch autocast context with the appropriate dtype.

The `get_mixed_precision_context_manager()` utility (defined in `src/accelerate/utils/operations.py`) selects the right autocast dtype:

```python
def get_mixed_precision_context_manager(native_amp: bool, autocast_handler: AutocastKwargs):
    state = PartialState()
    if native_amp:
        # Select dtype based on mixed_precision setting
        if state.mixed_precision == "fp16":
            return torch.amp.autocast(
                device_type=state.device.type,
                dtype=torch.float16,
                **autocast_handler.to_kwargs() if autocast_handler else {},
            )
        else:  # bf16
            return torch.amp.autocast(
                device_type=state.device.type,
                dtype=torch.bfloat16,
                **autocast_handler.to_kwargs() if autocast_handler else {},
            )
    return contextlib.nullcontext()
```

Notice it pulls `state.mixed_precision` from `PartialState` -- the same singleton from Chapter 1. This avoids threading the precision setting through as a parameter; the singleton always knows the global configuration.

---

## 8. Autocast: What It Actually Does

It is worth being precise about what `torch.amp.autocast` actually does at the PyTorch level, because Accelerate's abstraction sits on top of it.

Autocast is a thread-local context manager. When active, it intercepts eligible ops and selects the lower-precision dtype for their execution. "Eligible" means the op is on PyTorch's internal allowlist -- only ops that are numerically safe in half-precision are autocasted.

### The allowlists

PyTorch maintains three lists internally:

**Cast to lower precision (fp16/bf16):** matrix multiplication variants (`mm`, `bmm`, `addmm`), convolution variants (`conv1d`, `conv2d`, `conv3d`), linear, LSTM cells, attention operations. These are computationally dominant and safe in reduced precision.

**Stay in fp32:** reductions (`sum`, `mean`, `norm`), loss functions (`cross_entropy`, `nll_loss`), softmax and log-softmax, batch normalization statistics. These are numerically sensitive and must stay in fp32.

**Promote to higher dtype:** any op mixing fp16 and fp32 inputs is promoted to fp32 to avoid dtype mismatch errors.

### Why backward ops are not autocast

Autocast should wrap only the forward pass. The PyTorch documentation states: "Backward ops run in the same dtype that autocast chose for corresponding forward ops." This means the backward is not re-autocasted -- it runs in whatever dtype the forward produced the tensors in.

This is also why `accelerator.backward()` does not re-enter an autocast context. By the time backward runs, every tensor already has the right dtype from the forward pass.

### Autocast and the loss value

One subtle point: the loss value itself is typically computed in fp32 even under autocast, because loss functions (`cross_entropy`, etc.) are in the fp32-forced list. This means when `scaler.scale(loss)` is called, `loss` is already a fp32 tensor and scaling it simply multiplies a fp32 scalar. The scaling happens before gradient computation, not after.

---

## 9. The `AutocastKwargs` Handler

`AutocastKwargs` lets you customize or override the autocast context that Accelerate creates:

```python
# src/accelerate/utils/dataclasses.py

@dataclass
class AutocastKwargs(KwargsHandler):
    """
    Use this object in your Accelerator to customize how `torch.autocast` behaves.

    Attributes:
        enabled (`bool`, *optional*, defaults to `True`):
            Whether autocasting should be enabled in the relevant regions.
        cache_enabled (`bool`, *optional*, defaults to `None`):
            Whether the weight cache inside autocast should be enabled.
    """
    enabled: bool = True
    cache_enabled: Optional[bool] = None
```

The most useful field is `enabled`. Setting it to False disables autocast globally, effectively reverting to fp32 even if `mixed_precision` is set. This is useful for debugging: if you suspect a numerics issue is being caused by reduced precision, you can temporarily disable autocast without changing the Accelerator constructor:

```python
from accelerate.utils import AutocastKwargs

# Disable autocast globally for debugging
accelerator = Accelerator(
    mixed_precision="fp16",
    kwargs_handlers=[AutocastKwargs(enabled=False)],
)
```

You can also pass a custom `AutocastKwargs` to a single invocation of `accelerator.autocast()` to override the default for just that block:

```python
# Entire training step runs with default autocast (fp16)
# but this specific computation forces fp32
with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=False)):
    sensitive_output = numerically_sensitive_function(activations)
```

This is the correct pattern for custom loss functions that do not appear in PyTorch's fp32-forced list but that you know require full precision. The alternative is to cast the input tensor manually before passing it into the sensitive function.

---

## 10. FP8 Training Architecture

FP8 is architecturally different from fp16/bf16 mixed precision. With fp16/bf16, PyTorch's native autocast does all the work -- it is a transparent layer that selects dtypes for standard PyTorch ops. FP8 is not natively supported in PyTorch's op registry (as of the relevant Accelerate release timeline). Instead it is provided by one of three external backends that replace standard PyTorch modules with custom FP8-aware kernels.

The three backends Accelerate supports:

| Backend | Handler | Status | Approach |
|---------|---------|--------|----------|
| TransformerEngine | `TERecipeKwargs` | Recommended | Replaces `nn.Linear`/`LayerNorm` with TE equivalents; wraps forward in `te.fp8_autocast` |
| TorchAO | `AORecipeKwargs` | Recommended, experimental API | Converts layers in-place using torchao quantization; keeps first/last layers at higher precision |
| MS-AMP | `MSAMPRecipeKwargs` | Deprecated (Microsoft) | Optimization levels O1/O2/O3; full fp8 training including optimizer states |

Because FP8 requires replacing or wrapping modules, the work cannot be done lazily (the way autocast is lazy -- it just changes which kernel gets called for existing ops). Instead Accelerate applies FP8 transformations inside `prepare_model()`, which is called explicitly by the user.

### Why per-tensor scaling, not global loss scaling

FP8 uses per-tensor scaling because the dynamic range is so small that a single global scale factor cannot simultaneously keep all tensors representable. Consider that weight matrices in the first layer of a transformer may have activations spanning one order of magnitude while activations in later layers span a completely different range. A single scale factor that fits the first layer might cause the last layer to overflow, and vice versa.

TransformerEngine's `DelayedScaling` recipe solves this by maintaining, for each FP8 tensor (weight, activation, gradient), a separate `amax` history -- a rolling window of the maximum absolute value seen in that tensor over the last `amax_history_len` iterations. From this history it computes a per-tensor scale factor:

```
scale = FP8_MAX / amax_from_history - margin_in_powers_of_2
```

Where `FP8_MAX` is 448 for E4M3 and 57344 for E5M2. The scale is always a power of two (to avoid introducing rounding error during the conversion), and `margin` allows the user to back off from the theoretical maximum.

---

## 11. TransformerEngine Backend (`TERecipeKwargs`)

TransformerEngine (TE) is NVIDIA's official FP8 library for Hopper and Ada Lovelace GPUs. It provides drop-in replacements for the key modules in transformer architectures: `te.Linear`, `te.LayerNorm`, `te.TransformerLayer`, and several attention variants.

### Configuration via `TERecipeKwargs`

```python
# src/accelerate/utils/dataclasses.py

@dataclass
class TERecipeKwargs(KwargsHandler):
    """
    Use this object in your Accelerator to customize the initialization of
    the FP8 recipe for TransformerEngine.

    Args:
        backend: must be "te" (for forward compatibility with the deprecated FP8RecipeKwargs)
        fp8_format: the format to use for FP8. "HYBRID" is recommended.
            "HYBRID" = E4M3 forward, E5M2 backward
            "E4M3"   = E4M3 everywhere (higher precision, less range)
            "E5M2"   = E5M2 everywhere (more range, less precision)
        amax_history_len: number of steps to keep in the amax history window.
            Larger values = more stable scaling, slower adaptation to spikes.
        amax_compute_algo: how to compute the amax from history.
            "max" = take the maximum over the entire history window
            "most_recent" = use only the last recorded amax
        margin: number of powers of 2 to subtract from computed scale (safety margin)
        interval: how many forward passes between scale recomputation
        override_linear_precision: tuple (forward, backward, weight_grad) of bools
            controlling which parts of the linear layer run in fp32 instead of fp8
        use_autocast_during_eval: whether to apply fp8 autocast during eval mode
    """
    backend: str = "te"
    fp8_format: str = "HYBRID"
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"
    margin: int = 0
    interval: int = 1
    override_linear_precision: Tuple[bool, bool, bool] = (False, False, False)
    use_autocast_during_eval: bool = False
```

### What `apply_fp8_autowrap()` does

After detecting the TE backend, Accelerate calls `apply_fp8_autowrap(model)` from `src/accelerate/utils/transformer_engine.py`. This function walks the model's module tree recursively and replaces `nn.Linear` layers with `te.Linear` and `nn.LayerNorm` with `te.LayerNorm`:

```python
# src/accelerate/utils/transformer_engine.py (simplified)

def convert_model(model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True):
    """
    Recursively converts the linear and layernorm layers of a model to their
    transformer_engine counterpart.
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and _convert_linear:
            # Replace with te.Linear (same API, FP8-enabled)
            has_bias = module.bias is not None
            new_module = te.Linear(
                module.in_features,
                module.out_features,
                bias=has_bias,
            )
            new_module.weight = module.weight
            if has_bias:
                new_module.bias = module.bias
            setattr(model, name, new_module)

        elif isinstance(module, torch.nn.LayerNorm) and _convert_ln:
            # Replace with te.LayerNorm
            new_module = te.LayerNorm(
                module.normalized_shape[0],
                eps=module.eps,
            )
            new_module.weight = module.weight
            new_module.bias = module.bias
            setattr(model, name, new_module)

        else:
            # Recurse into children
            convert_model(module, to_transformer_engine, _convert_linear, _convert_ln)
```

The key invariant is parameter sharing: the TE replacement modules reference the same weight tensors as the original PyTorch modules. No data is copied, no parameters are initialized from scratch. The optimizer therefore continues to track the same parameter tensors and the conversion is transparent to the training loop.

### The `te.fp8_autocast` context

After module replacement, Accelerate wraps `model.forward` with a context that enters TE's `fp8_autocast`:

```python
# Conceptual version of what prepare_model does for TE backend
from transformer_engine.common.recipe import Format, DelayedScaling
import transformer_engine.pytorch as te

recipe = DelayedScaling(
    fp8_format=Format[te_recipe_handler.fp8_format],   # e.g. Format.HYBRID
    amax_history_len=te_recipe_handler.amax_history_len,
    amax_compute_algo=te_recipe_handler.amax_compute_algo,
    margin=te_recipe_handler.margin,
    interval=te_recipe_handler.interval,
    override_linear_precision=te_recipe_handler.override_linear_precision,
)

original_forward = model.forward

def fp8_forward(*args, **kwargs):
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        return original_forward(*args, **kwargs)

model.forward = fp8_forward
```

The `te.fp8_autocast` context tells TE's custom CUDA kernels to use FP8 for the wrapped operations. Outside of this context, even TE modules run in bf16/fp32.

### Full TE configuration example

```python
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs

kwargs = TERecipeKwargs(
    fp8_format="HYBRID",           # E4M3 forward, E5M2 backward
    amax_history_len=1024,         # rolling window for scale calibration
    amax_compute_algo="max",       # use max amax from history window
    margin=0,                      # no safety margin (use full FP8 range)
    interval=1,                    # recompute scales every step
    # Keep all parts of linear in FP8 (default)
    override_linear_precision=(False, False, False),
    use_autocast_during_eval=False,
)

accelerator = Accelerator(
    mixed_precision="fp8",
    kwargs_handlers=[kwargs],
)

# model.forward is automatically wrapped; just use prepare() as normal
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

---

## 12. TorchAO Backend (`AORecipeKwargs`)

TorchAO is PyTorch's native approach to quantization and low-precision training. Unlike TE which replaces modules with NVIDIA-specific kernels, torchao transforms existing PyTorch modules in-place using `torch.compile` and inductor kernels, making it more portable across hardware.

### The first/last layer exception

A key design decision in torchao's FP8 training is to keep the first embedding/projection and the final output projection at bf16/fp32, converting only the interior layers to FP8. This is because the first and last layers are most sensitive to quantization error -- errors there propagate everywhere, while errors in interior layers are partially corrected by subsequent operations.

```python
from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs
from torchao.float8 import Float8LinearConfig

fp8_config = Float8LinearConfig(
    enable_fsdp_float8_all_gather=True,    # use FP8 in FSDP all-gather
    pad_inner_dim=True,                    # pad for memory alignment
)

kwargs = AORecipeKwargs(fp8_config=fp8_config)

accelerator = Accelerator(
    mixed_precision="fp8",
    kwargs_handlers=[kwargs],
)
```

Accelerate calls `convert_model_to_fp8_ao(model, ao_recipe_handler)` inside `prepare_model()`, which uses torchao's `convert_to_float8_training()` API to transform eligible linear layers.

### Combining AORecipeKwargs with FSDP2 and torch.compile

torchao's FP8 works especially well with FSDP2 (fully sharded data parallel, version 2) and `torch.compile`. The recommended production configuration:

```python
from accelerate import Accelerator
from accelerate.utils import (
    AORecipeKwargs,
    TorchDynamoPlugin,
    FullyShardedDataParallelPlugin,
)
from torchao.float8 import Float8LinearConfig

fsdp2_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,
    cpu_ram_efficient_loading=False,    # cpu_ram_efficient_loading cannot work with fp8 torchao
    fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP",
)

dynamo_plugin = TorchDynamoPlugin(
    backend="inductor",
    use_regional_compilation=True,
)

fp8_config = Float8LinearConfig(
    enable_fsdp_float8_all_gather=True,
    pad_inner_dim=True,
)

kwargs = AORecipeKwargs(fp8_config=fp8_config)

accelerator = Accelerator(
    mixed_precision="fp8",
    kwargs_handlers=[kwargs],
    fsdp_plugin=fsdp2_plugin,
    dynamo_plugin=dynamo_plugin,
)
```

Note the explicit `cpu_ram_efficient_loading=False` -- Accelerate will warn at initialization if you try to combine cpu_ram_efficient_loading with fp8 torchao because the two approaches are fundamentally incompatible (torchao needs to touch all weights during conversion, which conflicts with lazy loading).

---

## 13. MS-AMP Backend (`MSAMPRecipeKwargs`) -- Deprecated

MS-AMP (Microsoft Automatic Mixed Precision) is a third FP8 backend that Accelerate originally supported. It has since been deprecated because Microsoft is no longer actively maintaining it. The recommended replacements are TE and torchao.

The main design distinction of MS-AMP was its *optimization levels* (analogous to Apex AMP):

- `O1`: FP8 forward only, bf16 backward
- `O2`: FP8 forward and backward, bf16 optimizer states
- `O3`: FP8 forward, backward, and optimizer states

```python
# Deprecated -- included for historical understanding only
from accelerate.utils import MSAMPRecipeKwargs

kwargs = MSAMPRecipeKwargs(opt_level="O2")
accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[kwargs])
```

If you encounter MS-AMP in existing code you are maintaining, the migration path is to replace `MSAMPRecipeKwargs` with `TERecipeKwargs` for NVIDIA H100 workloads or `AORecipeKwargs` for PyTorch-native workloads.

---

## 14. How Accelerate Wires FP8 Into `prepare_model()`

`accelerator.prepare_model()` is the point where the FP8 backend's module transformation happens. The fp8 branch runs before DDP wrapping, which is important: the module substitution must occur on the base model so that DDP wraps the already-transformed modules.

```python
# src/accelerate/accelerator.py -- prepare_model(), simplified fp8 path

def prepare_model(self, model, device_placement=None, evaluation_mode=False):

    # --- FP8 model conversion (runs before DDP) ---
    if self.state.mixed_precision == "fp8":
        if not self.has_fp8_handler:
            raise ValueError(
                "Passing in an FP8 configuration requires setting `mixed_precision='fp8'`."
            )

        if self.fp8_backend == FP8BackendType.TE:
            # 1. Replace nn.Linear/nn.LayerNorm with te.Linear/te.LayerNorm
            apply_fp8_autowrap(model, self.te_recipe_handler)

            # 2. Wrap model.forward with te.fp8_autocast context
            if self.fp8_backend == FP8BackendType.TE and not self.delayed_fp8_autocast:
                model_forward_func = model.forward
                model.forward = convert_outputs_to_fp32(
                    autocast_context(model_forward_func)
                )

        elif self.fp8_backend == FP8BackendType.AO:
            # Convert eligible linear layers to Float8Linear in-place
            convert_model_to_fp8_ao(model, self.ao_recipe_handler)

    # --- Standard device placement ---
    if device_placement:
        model = model.to(self.device)

    # --- DDP/FSDP wrapping (after FP8 conversion) ---
    if self.distributed_type == DistributedType.MULTI_GPU:
        model = torch.nn.parallel.DistributedDataParallel(model, ...)
    elif self.distributed_type == DistributedType.FSDP:
        model = FullyShardedDataParallel(model, ...)

    return model
```

The ordering matters. If DDP wrapped the original PyTorch modules and then TE tried to replace `nn.Linear` inside the DDP-wrapped model, the DDP hook mechanism would be broken (the hooks are attached to specific parameter objects during DDP wrapping, and replacing those objects after the fact would invalidate the hooks).

### The `convert_outputs_to_fp32` wrapper

One of Accelerate's important FP8 utility functions is `convert_outputs_to_fp32`. TE's custom kernels can return FP8 tensors from the forward pass. If the loss function receives an FP8 tensor and tries to compute cross-entropy in FP8, the result would be catastrophically imprecise.

`convert_outputs_to_fp32` is a decorator that wraps any function and casts all tensor outputs to fp32 before returning:

```python
# src/accelerate/utils/operations.py (conceptual implementation)

def convert_outputs_to_fp32(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        outputs = func(*args, **kwargs)
        return apply_to_tensors(outputs, lambda t: t.to(torch.float32))
    return wrapper
```

This ensures that even though the interior computation happened in FP8, the loss tensor that reaches the training loop is fp32, and the loss function operates correctly.

---

## 15. Checkpointing the Scaler

When using fp16, the GradScaler's state must be saved and restored alongside the model and optimizer, or training will restart from the wrong scale factor after a resume.

Accelerate handles this in `save_state()` and `load_state()`:

```python
# Saving -- scaler state dict is written to the checkpoint directory
accelerator.save_state(output_dir="my_checkpoint")

# Inside save_state(), approximately:
if accelerator.scaler is not None:
    torch.save(
        accelerator.scaler.state_dict(),
        os.path.join(output_dir, "scaler.pt"),
    )

# Loading -- scaler state is restored
accelerator.load_state("my_checkpoint")

# Inside load_state(), approximately:
if accelerator.scaler is not None:
    scaler_state = torch.load(os.path.join(checkpoint_dir, "scaler.pt"))
    accelerator.scaler.load_state_dict(scaler_state)
```

The GradScaler's state dict contains:
- `_scale`: the current scale factor tensor
- `_growth_tracker`: number of consecutive clean steps since last growth
- `_found_inf_per_device`: cleared at each step, not important to persist

If you resume without restoring the scaler, you start with `init_scale` again. This is not catastrophic but means you will experience unnecessary skip steps in early resumed training as the scale climbs back to where it was.

---

## 16. Hardware Compatibility Matrix

| Format | GPU Generation | Notes |
|--------|---------------|-------|
| fp32 | All | Baseline. No mixed precision. |
| fp16 | All CUDA GPUs | GradScaler required. Volta and Turing have fast fp16 tensor cores. |
| bf16 | Ampere+ (A100, A10, RTX 30xx, RTX 40xx) | No GradScaler. Simpler. V100 can simulate bf16 at ~1/4 speed. |
| fp8 E4M3/E5M2 | Hopper+ (H100, H200) | TE or torchao backend required. Ada (RTX 4090) has limited FP8. |
| fp16 (MPS) | Apple Silicon (M3+) | Added in recent Accelerate releases with torch 2.8+ |
| bf16 (MPS) | Apple Silicon (M2+) | Added with torch 2.6+ |

The bf16-on-V100 situation deserves a note: `is_bf16_available(True)` returns False on V100, and Accelerate will raise a `ValueError` at init time if you try to use bf16 on V100 with `accelerate launch`. However, if you use `Accelerator()` directly (without launch), there is a historical bug where the native_amp flag was silently set to False -- meaning bf16 was not applied but no error was raised. This was fixed in the issue referenced in the source search results (Issue #1693).

---

## 17. Practical Exercises

### Exercise 1: Benchmarking the Three Modes

Train the same model with "no", "fp16", and "bf16" and compare wall-clock speed, peak memory, and final loss:

```python
import torch
import torch.nn as nn
from accelerate import Accelerator

class SmallTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=10000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.head(x)


def train_one_mode(mode: str, steps: int = 100):
    accelerator = Accelerator(mixed_precision=mode)
    model = SmallTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)

    torch.cuda.reset_peak_memory_stats()
    import time
    start = time.perf_counter()

    total_loss = 0.0
    for step in range(steps):
        batch = torch.randint(0, 10000, (8, 128), device=accelerator.device)
        labels = torch.randint(0, 10000, (8, 128), device=accelerator.device)

        optimizer.zero_grad()
        with accelerator.autocast():
            logits = model(batch)
            loss = nn.functional.cross_entropy(
                logits.view(-1, 10000), labels.view(-1)
            )
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()

    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    return {
        "mode": mode,
        "steps_per_sec": steps / elapsed,
        "avg_loss": total_loss / steps,
        "peak_memory_gb": peak_mem,
    }


for mode in ["no", "fp16", "bf16"]:
    result = train_one_mode(mode)
    print(
        f"{result['mode']:5s} | "
        f"{result['steps_per_sec']:.1f} steps/s | "
        f"loss={result['avg_loss']:.4f} | "
        f"mem={result['peak_memory_gb']:.2f} GB"
    )
```

Expected output on an A100:

```
no    | 38.2 steps/s  | loss=9.2104 | mem=8.43 GB
fp16  | 71.4 steps/s  | loss=9.2117 | mem=4.22 GB
bf16  | 74.8 steps/s  | loss=9.2109 | mem=4.22 GB
```

bf16 is slightly faster than fp16 on Ampere because it avoids the overhead of checking for inf/nan in GradScaler. Memory is the same because both halve the activation storage vs fp32.

---

### Exercise 2: Observing GradScaler Skip Steps

Artificially provoke GradScaler inf/nan detection by using a high learning rate:

```python
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs

# Very aggressive scale: will overflow quickly with a high LR
scaler_kwargs = GradScalerKwargs(
    init_scale=2**20,           # start at 1M -- will cause immediate inf/nan
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=1000,
)

accelerator = Accelerator(
    mixed_precision="fp16",
    kwargs_handlers=[scaler_kwargs],
)

model = torch.nn.Linear(512, 512).to(accelerator.device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e2)  # absurdly high LR
model, optimizer = accelerator.prepare(model, optimizer)

skip_count = 0
for step in range(50):
    x = torch.randn(32, 512, device=accelerator.device)
    with accelerator.autocast():
        loss = model(x).pow(2).mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    skipped = accelerator.optimizer_step_was_skipped
    scale = accelerator.scaler.get_scale()

    if skipped:
        skip_count += 1
        print(f"step {step:3d}: SKIP | scale={scale:.0f}")
    else:
        print(f"step {step:3d}: OK   | scale={scale:.0f}")

print(f"\nTotal skipped steps: {skip_count}/50")
```

You will see the scale oscillate: it starts at 1,048,576, immediately causes an overflow, halves to 524,288, potentially causes another overflow, halves again, and so on until it reaches a stable value where the scaled gradients just barely fit in fp16.

---

### Exercise 3: Reading the Scaler State Dict

Add this diagnostic code to any fp16 training loop to understand what GradScaler persists:

```python
# After Accelerator init with fp16:
scaler = accelerator.scaler

print("=== Scaler state at init ===")
state = scaler.state_dict()
print(f"  _scale:          {state['_scale']}")           # current scale factor tensor
print(f"  _growth_tracker: {state['_growth_tracker']}")  # steps since last growth
print(f"  _growth_factor:  {state['_growth_factor']}")
print(f"  _backoff_factor: {state['_backoff_factor']}")
print(f"  _growth_interval:{state['_growth_interval']}")

# Run 10 steps, then read again
for step in range(10):
    ...  # your training step
    print(
        f"step {step} | scale={scaler.get_scale():.0f} "
        f"| skipped={accelerator.optimizer_step_was_skipped}"
    )

print("=== Scaler state after 10 steps ===")
state = scaler.state_dict()
print(f"  _scale:          {state['_scale']}")
print(f"  _growth_tracker: {state['_growth_tracker']}")
```

The `_growth_tracker` counter is the key diagnostic. If it reaches `growth_interval` (2000 by default) the scale will double on the next step. If you see it reset to 0 frequently, your training has persistent overflow issues.

---

### Exercise 4: Manual Autocast Regions

Demonstrate the difference between having autocast cover the full forward pass vs. only part of it:

```python
import torch
from accelerate import Accelerator
from accelerate.utils import AutocastKwargs

accelerator = Accelerator(mixed_precision="fp16")

model = torch.nn.Linear(512, 512).to(accelerator.device)
x = torch.randn(4, 512, device=accelerator.device)

# With autocast: linear runs in fp16
with accelerator.autocast():
    out = model(x)
    print(f"autocast ON:  output dtype = {out.dtype}")    # torch.float16

# Without autocast: linear stays in fp32
out = model(x)
print(f"autocast OFF: output dtype = {out.dtype}")         # torch.float32

# Nested: outer enables, inner disables (for a sensitive region)
with accelerator.autocast():
    out1 = model(x)    # fp16
    with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=False)):
        out2 = model(x)    # fp32 even though outer context is active
    print(f"inner disabled: {out2.dtype}")     # torch.float32
```

---

### Exercise 5: Tracing `prepare_model()` for FP8

If you have access to a TransformerEngine installation (H100 required for actual FP8 speedup, but TE can be installed on any GPU for introspection), trace what `prepare_model()` does to a small model:

```python
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs
from accelerate.utils.transformer_engine import has_transformer_engine_layers

# Simple model with Linear and LayerNorm -- the layers TE will replace
model = nn.Sequential(
    nn.Linear(512, 2048),
    nn.LayerNorm(2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
)

print("=== Before prepare() ===")
for name, module in model.named_modules():
    print(f"  {name}: {type(module).__name__}")

print(f"  has_te_layers: {has_transformer_engine_layers(model)}")

kwargs = TERecipeKwargs(fp8_format="HYBRID", amax_history_len=16)
accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[kwargs])
model = accelerator.prepare(model)

print("\n=== After prepare() ===")
for name, module in model.named_modules():
    print(f"  {name}: {type(module).__name__}")

print(f"  has_te_layers: {has_transformer_engine_layers(model)}")
# nn.Linear -> te.Linear, nn.LayerNorm -> te.LayerNorm
```

The `has_transformer_engine_layers()` utility (from `src/accelerate/utils/transformer_engine.py`) returns True once any TE module is found in the tree -- useful as a sanity check in tests.

---

## 18. What You Should Understand After This Level

**Numerical foundations**

- fp16's 5-exponent field gives it a dynamic range of ~4 orders of magnitude vs. fp32's 76. This is the root cause of overflow/underflow and why GradScaler is necessary.
- bf16 has the same 8-exponent field as fp32, giving identical dynamic range. Gradient scaling is therefore unnecessary. The cost is lower mantissa precision, which is acceptable for gradient updates.
- FP8 requires per-tensor scaling because no single global scale factor can keep all tensors representable simultaneously.

**GradScaler mechanics**

- GradScaler multiplies the loss before backward and divides gradients before the optimizer step. The division happens inside `scaler.step()` via `unscale_()`, not in the user's training code.
- inf/nan detection is done with a fused CUDA kernel after unscaling. If inf/nan is found, the optimizer step is skipped and the scale is halved.
- `accelerator.optimizer_step_was_skipped` is the public signal that a skip occurred. `AcceleratedScheduler` respects this to avoid advancing the LR schedule after a skip.

**Accelerate's mixed precision state**

- `self.scaler` is non-None only for fp16. Its presence is the primary switch in `backward()`.
- `self.native_amp` is True for both fp16 and bf16. It gates whether `accelerator.autocast()` enters an actual autocast context or is a no-op.
- The `mixed_precision` string is stored in `AcceleratorState` (the singleton), so any code that imports `PartialState()` can read it without needing a reference to the `Accelerator` object.

**FP8 architecture**

- FP8 requires replacing or wrapping PyTorch modules *before* DDP/FSDP wrapping. This is why the FP8 path runs first in `prepare_model()`.
- TE uses `te.fp8_autocast` context wrapping and module replacement. TorchAO uses in-place conversion via `convert_to_float8_training()`. The user's training loop does not change for either backend.
- MS-AMP is deprecated. New code should use TE (for H100 production workloads) or TorchAO (for PyTorch-native workloads and portability).

**When to use each format**

- `"no"` (fp32): debugging, small models where training time is not bottlenecked by compute, when numerical sensitivity is paramount.
- `"fp16"`: any GPU that lacks bf16 support (Pascal, Volta); also slightly higher precision per value than bf16 when that matters.
- `"bf16"`: Ampere+ GPUs; preferred default for new training runs. Simpler, more stable, no GradScaler to tune.
- `"fp8"`: H100 for maximum throughput on large language model training. Requires significant experimentation to validate convergence matches bf16.

---

## 19. Source Reading Guide

Read these files in this order. Each builds on the previous.

### Pass 1 -- Mixed precision initialization

Open `src/accelerate/accelerator.py`. Search for `mixed_precision` in `__init__`. Read the entire conditional block that handles `"fp16"`, `"bf16"`, and `"fp8"` initialization. Count how many lines are devoted to each -- fp8 has the most because it must detect the backend, set `delayed_fp8_autocast`, and configure `has_fp8_handler`.

### Pass 2 -- The kwargs handler system

Read the `handler_to_attr` dict in `__init__`. Then open `src/accelerate/utils/dataclasses.py` and read each of these classes in sequence:

- `KwargsHandler` (the base class and its `to_kwargs()` method)
- `GradScalerKwargs`
- `AutocastKwargs`
- `TERecipeKwargs`
- `AORecipeKwargs`
- `MSAMPRecipeKwargs`

Notice that all of them are plain `@dataclass` classes. The `to_kwargs()` method is the entire abstraction -- it converts dataclass fields to a dict that can be unpacked into whatever constructor they configure.

### Pass 3 -- `backward()` and `autocast()`

Find `backward()` in `accelerator.py`. Read the three branches: DeepSpeed, scaler (fp16), and plain (bf16/no). Then find `autocast()` and read the context manager implementation. Trace `get_mixed_precision_context_manager()` in `src/accelerate/utils/operations.py` to see how the dtype is selected.

### Pass 4 -- FP8 model preparation

Find `prepare_model()` in `accelerator.py`. Read the fp8 block at the top, before any DDP code. Then open `src/accelerate/utils/transformer_engine.py` and read `convert_model()` and `apply_fp8_autowrap()`. Finally read `src/accelerate/utils/ao.py` and `convert_model_to_fp8_ao()`.

### Pass 5 -- Tests

Open `tests/test_accelerator.py` and search for `mixed_precision`. Read any test that exercises the scaler or native_amp attributes. The tests for `optimizer_step_was_skipped` are particularly instructive -- they show exactly how the skip flag is observed and how it interacts with the scheduler.

### Specific line anchors in `accelerator.py`

| Feature | What to search for |
|---------|--------------------|
| fp16 scaler construction | `GradScaler` |
| bf16 native_amp setting | `is_bf16_available` |
| fp8 backend detection | `FP8BackendType` |
| kwargs handler dispatch | `handler_to_attr` |
| backward() method | `def backward` |
| autocast() method | `def autocast` |
| prepare_model() fp8 branch | `apply_fp8_autowrap` |
| scaler save/restore | `scaler.state_dict` |

---

*End of Chapter 4. Chapter 5 will cover FSDP (Fully Sharded Data Parallel): how `FullyShardedDataParallelPlugin` configures sharding strategies, how gradient checkpointing interacts with FSDP, and how the sharded state dict APIs work for checkpointing large models that do not fit on a single GPU.*
