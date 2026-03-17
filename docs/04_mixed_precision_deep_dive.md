# Mixed Precision Training: A Deep Dive

## Table of Contents

1. [Why Mixed Precision?](#1-why-mixed-precision)
2. [Number Formats](#2-number-formats)
3. [The Core Idea: Master Weights](#3-the-core-idea-master-weights)
4. [How a Training Step Works](#4-how-a-training-step-works)
5. [Gradient Scaling (fp16 only)](#5-gradient-scaling-fp16-only)
6. [PyTorch's Autocast](#6-pytorchs-autocast)
7. [Mixed Precision in Distributed Training](#7-mixed-precision-in-distributed-training)
8. [FSDP1 vs FSDP2 Mixed Precision](#8-fsdp1-vs-fsdp2-mixed-precision)
9. [The Upcast Problem We Fixed](#9-the-upcast-problem-we-fixed)
10. [FP8 Training](#10-fp8-training)
11. [Memory Math](#11-memory-math)
12. [Common Pitfalls](#12-common-pitfalls)
13. [Decision Guide](#13-decision-guide)

---

## 1. Why Mixed Precision?

Two benefits: **speed** and **memory**.

```
┌─────────────────────────────────────────────────────────────────┐
│                   WHY MIXED PRECISION?                          │
├─────────────────────────┬───────────────────────────────────────┤
│       SPEED             │         MEMORY                        │
│                         │                                       │
│  fp16/bf16 ops are      │  Half the bytes per parameter:        │
│  2-8x faster on         │                                       │
│  modern GPUs (Tensor    │  fp32: 4 bytes/param                  │
│  Cores)                 │  fp16: 2 bytes/param                  │
│                         │  fp8:  1 byte/param                   │
│  Doubles throughput     │                                       │
│  for matmuls            │  Fits 2x larger models in same VRAM   │
└─────────────────────────┴───────────────────────────────────────┘
```

But lower precision loses information. The trick: use low precision where it's safe, fp32 where it matters.

---

## 2. Number Formats

### IEEE 754 Floating Point Layout

```
fp32 (32 bits):
┌──────┬──────────┬───────────────────────┐
│ sign │ exponent │       mantissa        │
│  1   │    8     │         23            │
└──────┴──────────┴───────────────────────┘
Range: ±3.4e38    Precision: ~7 decimal digits


fp16 (16 bits):
┌──────┬──────────┬───────────┐
│ sign │ exponent │ mantissa  │
│  1   │    5     │    10     │
└──────┴──────────┴───────────┘
Range: ±65504     Precision: ~3.3 decimal digits


bf16 (16 bits):
┌──────┬──────────┬─────────┐
│ sign │ exponent │mantissa │
│  1   │    8     │    7    │
└──────┴──────────┴─────────┘
Range: ±3.4e38    Precision: ~2.4 decimal digits


fp8 E4M3 (8 bits):
┌──────┬─────┬─────┐
│ sign │ exp │mant │
│  1   │  4  │  3  │
└──────┴─────┴─────┘
Range: ±448        Precision: ~1.7 decimal digits


fp8 E5M2 (8 bits):
┌──────┬─────┬───┐
│ sign │ exp │ m │
│  1   │  5  │ 2 │
└──────┴─────┴───┘
Range: ±57344     Precision: ~1.2 decimal digits
```

### Comparison Table

```
Format  │ Bytes │ Range       │ Precision     │ Grad Scaling? │ Hardware
────────┼───────┼─────────────┼───────────────┼───────────────┼──────────────
fp32    │   4   │ ±3.4e38     │ 7 digits      │ No            │ All
fp16    │   2   │ ±65504      │ 3.3 digits    │ YES           │ All GPUs
bf16    │   2   │ ±3.4e38     │ 2.4 digits    │ No            │ Ampere+ (A100)
fp8 E4M3│   1   │ ±448        │ 1.7 digits    │ Per-tensor    │ Hopper+ (H100)
fp8 E5M2│   1   │ ±57344      │ 1.2 digits    │ Per-tensor    │ Hopper+ (H100)
```

### Why fp16 Needs Gradient Scaling but bf16 Doesn't

```
                    Representable range
                    ◄──────────────────────────────────────────►

fp32:  |·······················|========================|·······|
       0                   1e-38                    3.4e38

bf16:  |·······················|========================|·······|
       0                   1e-38                    3.4e38
       (same exponent range as fp32, just less precision)

fp16:  |··|====================|
       0  1e-8             65504
       ▲
       │ Small gradients fall here → UNDERFLOW TO ZERO
       │ GradScaler multiplies loss to shift gradients into safe range
```

bf16 has the same exponent (8 bits) as fp32, so it can represent the same range of values. fp16 has only 5 exponent bits, so very small gradients (common in deep networks) underflow to zero.

---

## 3. The Core Idea: Master Weights

The fundamental principle of mixed precision:

> **Keep fp32 "master" copies of weights. Cast to low precision only for compute.**

```
┌─────────────────────────────────────────────────────────────┐
│                    PARAMETER STORAGE                         │
│                                                              │
│   ┌──────────────────┐                                       │
│   │  Master Weights  │  ◄── fp32 (4 bytes/param)            │
│   │  (optimizer sees │      Accumulated small updates        │
│   │   these)         │      maintain precision over          │
│   └────────┬─────────┘      thousands of steps               │
│            │                                                 │
│            │ cast to bf16/fp16                                │
│            ▼                                                 │
│   ┌──────────────────┐                                       │
│   │  Working Weights  │ ◄── bf16/fp16 (2 bytes/param)       │
│   │  (forward/backward│     Fast Tensor Core math            │
│   │   uses these)     │                                      │
│   └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### Why Master Weights Matter

Consider a parameter with value `1.0` and a gradient update of `0.0001`:

```
Without master weights (pure bf16):
  Step 1: 1.0 + 0.0001 = 1.0        ← bf16 can't represent this!
  Step 2: 1.0 + 0.0001 = 1.0        ← lost again
  ...1000 steps later: still 1.0    ← NO LEARNING

With master weights (fp32 master, bf16 compute):
  Step 1: 1.0000000 + 0.0001 = 1.0001000  (fp32 master updated)
  Step 2: 1.0001000 + 0.0001 = 1.0002000
  ...1000 steps later: 1.1000000           ← CORRECT LEARNING
  (cast to bf16 for next forward: 1.1016)
```

The key insight: **individual updates are small, but they accumulate**. fp32 preserves the accumulation; bf16 loses it.

---

## 4. How a Training Step Works

### Standard Mixed Precision (Single GPU)

```
                        FORWARD PASS
                        ════════════

   fp32 master weights ──cast──► bf16 weights
                                    │
   bf16 input ─────────────────────►│
                                    ▼
                              ┌──────────┐
                              │  MatMul   │ ◄── Tensor Cores (fast!)
                              │  (bf16)   │
                              └────┬─────┘
                                   │
                                   ▼
                              bf16 activations
                                   │
                                   ▼
                              ┌──────────┐
                              │ LayerNorm │ ◄── autocast keeps in fp32
                              │  (fp32)   │     (numerically sensitive)
                              └────┬─────┘
                                   │
                                   ▼
                              ┌──────────┐
                              │  Softmax  │ ◄── autocast keeps in fp32
                              │  (fp32)   │
                              └────┬─────┘
                                   │
                                   ▼
                              bf16 output ───► fp32 loss


                        BACKWARD PASS
                        ═════════════

   fp32 loss
       │
       ▼ (scaled by GradScaler if fp16)
   bf16 intermediate gradients ◄── computed in bf16 during
       │                            backward through autocast regions
       │
       ▼ accumulated into .grad
   fp32 gradients ◄── .grad lives on fp32 master params,
       │               so PyTorch accumulates in fp32
       │               (unscaled by GradScaler if fp16)
       ▼
   ┌───────────┐
   │ Optimizer  │
   │ (AdamW)    │ ◄── operates on fp32 master weights
   │            │     fp32 gradients + fp32 momentum/variance
   └─────┬─────┘
         │
         ▼
   fp32 master weights (updated)
```

### Which Ops Run in Which Precision?

PyTorch's autocast has three categories:

```
┌──────────────────────────────────────────────────────────────┐
│  ALWAYS LOW PRECISION (bf16/fp16)     — speed-critical       │
│  ─────────────────────────────────                           │
│  • torch.mm, torch.matmul, torch.bmm                        │
│  • torch.nn.functional.linear                                │
│  • torch.nn.functional.conv1d/2d/3d                          │
│  • torch.baddbmm                                             │
├──────────────────────────────────────────────────────────────┤
│  ALWAYS FP32                          — precision-critical   │
│  ──────────                                                  │
│  • torch.nn.functional.layer_norm                            │
│  • torch.nn.functional.softmax                               │
│  • torch.nn.functional.cross_entropy                         │
│  • torch.nn.functional.binary_cross_entropy                  │
│  • torch.nn.functional.batch_norm, group_norm                │
│  • torch.nn.functional.log_softmax                           │
├──────────────────────────────────────────────────────────────┤
│  MATCH INPUT DTYPE                    — follow the data      │
│  ────────────────                                            │
│  • torch.cat, torch.stack                                    │
│  • torch.nn.functional.relu                                  │
│  • torch.nn.functional.dropout                               │
│  • Element-wise ops (+, -, *, /)                             │
│  • torch.pow, torch.log, torch.exp                           │
│  • torch.sum, torch.mean (reductions)                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Gradient Scaling (fp16 only)

### The Underflow Problem

```
                     fp16 representable range
                     ┌─────────────────────────────────┐
                     │                                  │
   ──────────────────┼──────────────────────────────────┼──────
   0            5.96e-8                              65504
                     ▲
                     │
              Minimum positive fp16 value


   Gradient distribution in a deep network:

   Count
   ▲
   │
   │  ███
   │  ████
   │  █████
   │  ██████                           ▨▨  (few large grads)
   │  ████████                      ▨▨▨▨
   │  ██████████               ▨▨▨▨▨▨▨▨
   │  █████████████████▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨
   └──┼────────────────┼──────────────────────────► magnitude
      │                │
   UNDERFLOW        fp16 min
   (lost!)          5.96e-8

   Many gradients are smaller than fp16 min → they become 0 → no learning!
```

### How GradScaler Fixes It

```
Step 1: SCALE UP                          Step 2: BACKWARD
─────────────                             ────────────────
loss = model(input)                       scaled_loss.backward()
scaled_loss = loss * scale_factor         # All gradients are scale_factor × larger
              (e.g., 65536)               # Small grads that would underflow are now
                                          # in representable range!

Step 3: UNSCALE + CHECK                   Step 4: UPDATE SCALE
───────────────────────                   ──────────────────
optimizer.step()                          if no inf/nan for N steps:
# Internally divides grads by               scale_factor *= 2  (grow)
# scale_factor before updating            else:
# Checks for inf/nan                        scale_factor /= 2  (shrink)
# If inf/nan: SKIP this step                skip optimizer step
```

```
Scale factor over training:

scale
  ▲
  │
  │    ╱╲        ╱╲        ╱╲         ╱──────── (stabilizes)
  │   ╱  ╲      ╱  ╲      ╱  ╲       ╱
  │  ╱    ╲    ╱    ╲    ╱    ╲     ╱
  │ ╱      ╲  ╱      ╲  ╱      ╲   ╱
  │╱        ╲╱        ╲╱        ╲ ╱
  │                               ╲╱
  └──────────────────────────────────────────► training steps
     ▲         ▲          ▲
     │         │          │
  overflow!  overflow!   overflow!
  (halve)    (halve)    (halve)
```

### Why bf16 Doesn't Need This

bf16 has the same 8-bit exponent as fp32 → same dynamic range (±3.4e38). Small gradients that would underflow in fp16 are perfectly representable in bf16. The tradeoff: bf16 has less precision (7-bit mantissa vs fp16's 10-bit), but range matters more than precision for gradient accumulation.

---

## 6. PyTorch's Autocast

Autocast is the mechanism that selectively casts operations:

```python
# Without autocast — everything in fp32 (slow)
output = model(input)  # all ops in fp32

# With autocast — automatic per-op casting
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input)
    # Linear layers: bf16 (fast Tensor Core path)
    # LayerNorm: fp32 (precision-sensitive)
    # Softmax: fp32 (precision-sensitive)
    # ReLU: follows input dtype
```

### How Autocast Works Internally

```
┌────────────────────────────────────────────────────────────┐
│                    AUTOCAST DISPATCH                        │
│                                                             │
│  model.linear(x)                                            │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────┐                                    │
│  │ Is autocast enabled? │──No──► Run in original dtype      │
│  └──────────┬──────────┘                                    │
│             │ Yes                                           │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ Is op in fp16 list? │──Yes──► Cast inputs to bf16/fp16   │
│  │ (linear, matmul...) │        Run op in bf16/fp16         │
│  └──────────┬──────────┘                                    │
│             │ No                                            │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ Is op in fp32 list? │──Yes──► Cast inputs to fp32        │
│  │ (softmax, norm...)  │        Run op in fp32              │
│  └──────────┬──────────┘                                    │
│             │ No                                            │
│             ▼                                               │
│  Run op in widest input dtype (promote rule)                │
└────────────────────────────────────────────────────────────┘
```

### Accelerate's Autocast Integration

`accelerator.prepare(model)` **automatically wraps `model.forward`** with autocast + `convert_outputs_to_fp32`:

```python
# Inside accelerator.prepare() — you DON'T write this, it happens automatically:
if self.native_amp:
    autocast_context = get_mixed_precision_context_manager(...)
    model.forward = autocast_context(model.forward)          # wrap in autocast
    model.forward = convert_outputs_to_fp32(model.forward)   # cast outputs to fp32
```

So every `model(batch)` call already runs under autocast, and outputs are fp32:

```
model(batch)
   │
   ▼
autocast(dtype=bf16)         ◄── automatic from prepare()
   │
   ├── Linear layers: bf16   (fast Tensor Core math)
   ├── LayerNorm: fp32       (precision-sensitive)
   ├── Softmax: fp32         (precision-sensitive)
   │
   ▼
convert_outputs_to_fp32      ◄── automatic from prepare()
   │
   ▼
outputs in fp32              ◄── loss computed here is fp32 ✓
```

### When Do You Need `accelerator.autocast()` Explicitly?

Only for compute **outside** `model.forward()`:

```python
# Case 1: Simple — no explicit autocast needed
# model.forward already autocast-wrapped by prepare()
output = model(batch)           # ← autocast + convert_outputs_to_fp32
loss = criterion(output)        # ← fp32 (outputs already converted)
accelerator.backward(loss)

# Case 2: Custom compute outside model — USE accelerator.autocast()
output = model(batch)           # ← autocast from prepare()
with accelerator.autocast():
    # This projection is NOT inside model.forward,
    # so it needs its own autocast to run in bf16
    projected = custom_projection(output)
loss = criterion(projected.float())  # ← explicit cast to fp32 for loss
accelerator.backward(loss)
```

### Double-Wrapping is Safe

`torch.autocast` contexts are **reentrant and idempotent**:

```python
# This works — output is fp32 (convert_outputs_to_fp32 is baked into model.forward)
for batch in dataloader:
    with accelerator.autocast():          # outer autocast (redundant for model call)
        output = model(batch)             # output is fp32 regardless (convert_outputs_to_fp32)
        loss = criterion(output)          # depends on criterion type (see note below)

# Simpler and safer — keep loss outside autocast scope entirely
for batch in dataloader:
    output = model(batch)                 # autocast from prepare(), output is fp32
    loss = criterion(output)              # fp32 ✓ guaranteed
```

**Note on loss under autocast:** Standard loss functions (`nn.CrossEntropyLoss`, `nn.BCELoss`,
`nn.MSELoss`) are in autocast's fp32 promotion list — they run in fp32 even under autocast.
But custom losses with matmuls or other "low-precision-eligible" ops may silently run in bf16.
The safest approach: keep loss computation outside any autocast scope.

---

## 7. Mixed Precision in Distributed Training

### DDP (DistributedDataParallel)

Simple — each GPU has a full model copy. Autocast works locally per GPU.

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  GPU 0   │    │  GPU 1   │    │  GPU 2   │
│          │    │          │    │          │
│ fp32     │    │ fp32     │    │ fp32     │
│ master   │    │ master   │    │ master   │
│ weights  │    │ weights  │    │ weights  │
│    │     │    │    │     │    │    │     │
│    ▼     │    │    ▼     │    │    ▼     │
│ autocast │    │ autocast │    │ autocast │
│ bf16 fwd │    │ bf16 fwd │    │ bf16 fwd │
│    │     │    │    │     │    │    │     │
│    ▼     │    │    ▼     │    │    ▼     │
│ fp32 grad│    │ fp32 grad│    │ fp32 grad│
│    │     │    │    │     │    │    │     │
└────┼─────┘    └────┼─────┘    └────┼─────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              AllReduce (fp32)*
              Average gradients
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
  optimizer       optimizer       optimizer
  step (fp32)     step (fp32)     step (fp32)

  * Default DDP AllReduces gradients in the parameter dtype (fp32).
    Gradients are stored in .grad on fp32 master params, so they are fp32.
    To reduce in bf16 for bandwidth savings, install a communication hook:
    ddp_model.register_comm_hook(state=None, hook=bf16_compress_hook)
```

### FSDP (Fully Sharded Data Parallel)

More complex — weights are **sharded** across GPUs. Mixed precision has additional dtype conversions during all-gather and reduce-scatter.

```
┌──────────────────────────────────────────────────────────────────┐
│                    FSDP MIXED PRECISION                           │
│                                                                   │
│                    ┌────────────┐                                  │
│   GPU 0 has       │  Shard 0   │  fp32 (master weight shard)     │
│                    │  (1/N of   │                                  │
│                    │   params)  │                                  │
│                    └─────┬──────┘                                  │
│                          │                                        │
│        ┌─────────────────┼─────────────────┐                      │
│        │           ALL-GATHER              │                      │
│        │     Collect all shards from       │                      │
│        │     all GPUs → full param         │                      │
│        │                                   │                      │
│        │     MixedPrecisionPolicy.         │                      │
│        │     param_dtype controls          │                      │
│        │     the dtype AFTER gather        │                      │
│        └─────────────────┬─────────────────┘                      │
│                          │                                        │
│                          ▼                                        │
│                    ┌────────────┐                                  │
│                    │ Full param │  bf16 (param_dtype)              │
│                    │ (all N/N)  │  Used for forward/backward      │
│                    └─────┬──────┘                                  │
│                          │                                        │
│                     Forward + Backward                            │
│                          │                                        │
│                          ▼                                        │
│                    ┌────────────┐                                  │
│                    │ Gradients  │  bf16                            │
│                    └─────┬──────┘                                  │
│                          │                                        │
│        ┌─────────────────┼─────────────────┐                      │
│        │         REDUCE-SCATTER            │                      │
│        │     Each GPU gets 1/N of          │                      │
│        │     averaged gradients            │                      │
│        │                                   │                      │
│        │     MixedPrecisionPolicy.         │                      │
│        │     reduce_dtype controls         │                      │
│        │     the dtype for reduction       │                      │
│        └─────────────────┬─────────────────┘                      │
│                          │                                        │
│                          ▼                                        │
│                    ┌────────────┐                                  │
│                    │ Grad shard │  Cast back to fp32               │
│                    │ (1/N)      │  (matches master weight shard)   │
│                    └─────┬──────┘                                  │
│                          │                                        │
│                          ▼                                        │
│                    ┌────────────┐                                  │
│                    │ Optimizer  │  fp32 params + fp32 grads        │
│                    │ step       │  + fp32 momentum/variance        │
│                    └────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘
```

### MixedPrecisionPolicy Controls

```python
# FSDP2 API (torch >= 2.x). FSDP1 uses `MixedPrecision` (different class name).
from torch.distributed.fsdp import MixedPrecisionPolicy

policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # Cast params to bf16 during forward/backward
    reduce_dtype=torch.bfloat16,   # Gradient reduction in bf16
    output_dtype=torch.bfloat16,   # Module output dtype
)
```

```
Master weight (fp32)
       │
       ├── param_dtype ──► bf16 during forward/backward (all-gathered)
       │
       ├── reduce_dtype ──► bf16 during gradient reduce-scatter
       │
       └── after reduce ──► cast back to fp32 for optimizer step
```

---

## 8. FSDP1 vs FSDP2 Mixed Precision

### FSDP1: Explicit Dtype Tracking

```
FSDP1 wraps params in FlatParameter handles:

┌─────────────────────────────────────────────────┐
│  FlatParameter (FSDP1)                          │
│                                                  │
│  _handle._orig_param_dtype = torch.float32  ◄── explicitly stored
│  _handle._fwd_bwd_param_dtype = torch.bfloat16  │
│                                                  │
│  The handle KNOWS the master dtype and casts     │
│  accordingly during forward/backward.            │
│                                                  │
│  To upcast: change BOTH param.data AND           │
│  _handle._orig_param_dtype                       │
└─────────────────────────────────────────────────┘
```

### FSDP2: Dtype Snapshotted at `fully_shard()` Time

```
FSDP2 wraps params as DTensor via FSDPParam:

┌─────────────────────────────────────────────────┐
│  FSDPParam (FSDP2)                              │
│                                                  │
│  Snapshots param.dtype at fully_shard() time     │
│  No explicit _orig_param_dtype attribute         │
│                                                  │
│  If fully_shard sees bf16 → records bf16         │
│  If fully_shard sees fp32 → records fp32         │
│                                                  │
│  MixedPrecisionPolicy hooks:                     │
│    pre-forward:  cast master_dtype → param_dtype  │
│    post-backward: cast grad back to master_dtype  │
│                                                  │
│  ⚠ Changing param.data dtype AFTER fully_shard   │
│    does NOT update the snapshotted dtype!         │
└─────────────────────────────────────────────────┘
```

### Why This Matters for Upcasting

```
NON-RAM-EFFICIENT PATH (simple):
═════════════════════════════════

  1. Model loaded in bf16
  2. Upcast to fp32:  param.data = param.data.to(fp32)  ✓
  3. fully_shard() → sees fp32 → records fp32
  4. Forward: fp32 → bf16 (via param_dtype)
  5. Backward: grads in bf16
  6. Post-backward: grads cast to fp32 (master dtype)
  7. Optimizer: fp32 params + fp32 grads  ✓


RAM-EFFICIENT PATH (complex):
═════════════════════════════

  The model starts on CPU, is moved to meta device, then weights
  are broadcast from rank 0 via fsdp2_load_full_state_dict.

  Timeline:
  ─────────

  1. Model loaded in bf16 on CPU (rank 0 only has real weights)

  2. State dict saved: original_sd = model.state_dict()  [bf16]

  3. ★ Upcast state dict to fp32
     ★ Move model to meta with dtype=fp32
     (both must agree for broadcast to work)

  4. fully_shard() → sees fp32 meta tensors → records fp32  ✓

  5. fsdp2_load_full_state_dict():
     - Rank 0: broadcasts fp32 state dict values
     - Other ranks: allocate fp32 receive buffers
       (size = sharded_param.dtype = fp32 from meta model)
     - Broadcast: fp32 ↔ fp32  ✓

  6. Forward: fp32 → bf16 (via param_dtype)
  7. Backward: grads in bf16
  8. Post-backward: grads cast to fp32  ✓
  9. Optimizer: fp32 params + fp32 grads  ✓
```

---

## 9. The Upcast Problem We Fixed

### Bug 1: Silent No-Op (Original Code on `main`)

```python
# WRONG — rebinds local variable, model unchanged
for name, param in model.named_parameters():
    if param.requires_grad and param.dtype != torch.float32:
        param = param.to(torch.float32)  # ← only changes local 'param'!
```

```
What happens in memory:

Before:
  model.weight ──────► Tensor(bf16, data=0x1000)
  param ─────────────► Tensor(bf16, data=0x1000)  (same object)

After param = param.to(torch.float32):
  model.weight ──────► Tensor(bf16, data=0x1000)  ← UNCHANGED!
  param ─────────────► Tensor(fp32, data=0x2000)  ← new object, thrown away
```

```python
# CORRECT — mutates the actual parameter tensor
for name, param in model.named_parameters():
    if param.requires_grad and param.dtype != torch.float32:
        param.data = param.data.to(torch.float32)  # ← mutates in-place
```

```
What happens in memory:

Before:
  model.weight ──────► Tensor(bf16, data=0x1000)
  param ─────────────► Tensor(bf16, data=0x1000)  (same object)

After param.data = param.data.to(torch.float32):
  model.weight ──────► Tensor(fp32, data=0x2000)  ← UPDATED!
  param ─────────────► Tensor(fp32, data=0x2000)  (still same object)
```

### Bug 2: Broadcast Dtype Mismatch (First Fix Attempt)

```
ATTEMPT: Upcast state dict to fp32, but leave meta model in bf16.

Rank 0 (has real weights):              Rank 1 (meta model):
┌──────────────────────┐               ┌──────────────────────┐
│ original_sd["weight"]│               │ sharded_param.dtype   │
│ = tensor(fp32)       │               │ = torch.bfloat16      │
│ nbytes = 400 bytes   │               │                       │
└──────────┬───────────┘               │ recv_buffer = empty(  │
           │                            │   dtype=bf16)         │
           │     broadcast              │ nbytes = 200 bytes   │
           │─────────────────────────►  └──────────────────────┘
           │                                      ▲
           │  fp32: 400 bytes                     │
           │  bf16: 200 bytes                     │
           │                                      │
           └──── MISMATCH! NCCL hangs/crashes ────┘
```

### The Correct Fix

```
Upcast BOTH state dict AND meta model to fp32:

Rank 0:                                 Rank 1:
┌──────────────────────┐               ┌──────────────────────┐
│ original_sd["weight"]│               │ sharded_param.dtype   │
│ = tensor(fp32)       │               │ = torch.float32  ✓   │
│ nbytes = 400 bytes   │               │                       │
└──────────┬───────────┘               │ recv_buffer = empty(  │
           │                            │   dtype=fp32)    ✓   │
           │     broadcast              │ nbytes = 400 bytes   │
           │─────────────────────────►  └──────────────────────┘
           │                                      ✓
           │  fp32: 400 bytes ══════ fp32: 400 bytes
           │
           └──── MATCH! Broadcast succeeds ─────────

  AND fully_shard() saw fp32 meta tensors → records fp32 master dtype
  → gradients cast back to fp32 → optimizer happy  ✓
```

---

## 10. FP8 Training

FP8 is the next frontier — 1 byte per element, but requires per-tensor scaling.

```
┌──────────────────────────────────────────────────────────────┐
│                    FP8 TRAINING LOOP                          │
│                                                               │
│  Two FP8 formats used together:                               │
│                                                               │
│  E4M3 (4-bit exp, 3-bit mantissa):                           │
│    • Used for FORWARD pass (weights + activations)            │
│    • Better precision (3-bit mantissa)                        │
│    • Range ±448                                               │
│                                                               │
│  E5M2 (5-bit exp, 2-bit mantissa):                           │
│    • Used for BACKWARD pass (gradients)                       │
│    • Better range (5-bit exponent)                            │
│    • Range ±57344                                             │
│                                                               │
│  Per-tensor scaling:                                          │
│    Each tensor gets its own scale factor to map values        │
│    into the representable FP8 range.                          │
│                                                               │
│    tensor_fp8 = tensor_fp32 * scale_factor                    │
│    scale_factor = max_fp8 / max(abs(tensor_fp32))             │
│                                                               │
│  Delayed scaling:                                             │
│    Scale factors are computed from PREVIOUS iteration's       │
│    statistics (avoids extra forward pass for calibration).    │
└──────────────────────────────────────────────────────────────┘
```

### FP8 in Accelerate

```python
# TransformerEngine backend
from accelerate.utils import TERecipeKwargs
accelerator = Accelerator(
    mixed_precision="fp8",
    kwargs_handlers=[TERecipeKwargs()]
)

# TorchAO backend
from accelerate.utils import AORecipeKwargs
accelerator = Accelerator(
    mixed_precision="fp8",
    kwargs_handlers=[AORecipeKwargs()]
)

# Note: FP8RecipeKwargs(backend="te") still works but is deprecated.
# Use TERecipeKwargs or AORecipeKwargs directly.
```

---

## 11. Memory Math

### Per-Parameter Memory Breakdown

```
                        fp32        Mixed (bf16)    Mixed (fp8)
                        ────        ────────────    ───────────
  Parameter             4 bytes     4 bytes*        4 bytes*
  Gradient              4 bytes     2 bytes         1 byte
  Optimizer (Adam):
    momentum            4 bytes     4 bytes         4 bytes
    variance            4 bytes     4 bytes         4 bytes
                        ───────     ───────         ───────
  TOTAL per param       16 bytes    14 bytes        13 bytes

  * Master weights always fp32 (4 bytes)

  Activations (forward):
    fp32: 4 bytes per activation
    bf16: 2 bytes per activation  ← biggest memory saving!
    fp8:  1 byte per activation
```

### Example: 7B Parameter Model

```
                    fp32 only       bf16 mixed      fp8 mixed
                    ─────────       ──────────      ─────────
  Parameters        28 GB           28 GB*          28 GB*
  Gradients         28 GB           14 GB           7 GB
  Adam states       56 GB           56 GB           56 GB
  ──────────────    ──────          ──────          ──────
  TOTAL             112 GB          98 GB           91 GB

  * Master weights stay fp32

  Activations (batch_size=8, seq_len=2048):
    fp32: ~40 GB
    bf16: ~20 GB     ← this is where mixed precision really saves
    fp8:  ~10 GB

  Total with activations:
    fp32: ~152 GB (needs 2x H100 80GB)
    bf16: ~118 GB (fits on 2x H100 80GB)
    fp8:  ~101 GB (more room for larger batches)
```

### With FSDP Sharding (N GPUs)

```
  Per-GPU memory ≈ (params + grads + optimizer_states) / N + activations

  7B model, 8x H100s, bf16 mixed:
    Sharded: (28 + 28 + 56) / 8 = 14 GB
    Activations: ~20 GB (not sharded by default)
    ──────────────────────
    Total per GPU: ~34 GB  ← fits in 80GB with room for large batches

  Note on gradient dtype with FSDP:
    During forward/backward, gradients are computed in bf16 (param_dtype).
    After reduce-scatter, gradient shards are cast BACK to fp32 (master dtype)
    for the optimizer step. So steady-state gradient memory per GPU is:
      fp32 grad shards = 7B × 4 bytes / 8 GPUs = 3.5 GB (not 1.75 GB)
    The bf16 gradients are transient (peak memory during backward only).
```

---

## 12. Common Pitfalls

### 1. Loss Should Be Computed in fp32, Not bf16

Loss values involve small numbers that get subtracted (e.g., log-probabilities in cross-entropy).
bf16 has only ~2.4 decimal digits of precision — small differences round to zero:

```
Example: cross-entropy on two similar logits

fp32: log(0.9998) - log(0.9997) = -0.0002000 - (-0.0003000) = 0.0001000  ✓

bf16: bf16(0.9998) = 1.0           ← rounded! (bf16 ULP at 1.0 is 0.0078125)
      bf16(0.9997) = 1.0           ← rounded to same value!
      log(1.0) - log(1.0) = 0.0    ← ZERO!

The precision is lost before log() even runs — both inputs collapse
to the same bf16 value. The difference vanishes.

Result: loss gradient vanishes → model stops learning
```

**The rule:** autocast the forward pass (heavy matmuls — speed matters, precision loss is tolerable),
compute loss in fp32 (tiny computation — speed gain is negligible, precision matters).

```python
# BAD — without prepare(), loss under autocast is computed in bf16
# (Only applies to raw PyTorch autocast without Accelerate's prepare())
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)  # bf16 loss! Precision lost.

# OK — with prepare(), output is ALWAYS fp32 thanks to convert_outputs_to_fp32
# Even inside an outer autocast, the output is fp32 because convert_outputs_to_fp32
# is the outermost wrapper: convert_outputs_to_fp32(autocast(original_forward))
with accelerator.autocast():
    output = model(input)             # output is fp32 (convert_outputs_to_fp32 is baked in)
    loss = criterion(output, target)  # ⚠ criterion runs under autocast though!
                                      # CrossEntropyLoss → fp32 (in autocast fp32 list) ✓
                                      # Custom matmul-based loss → bf16 ✗

# BEST — let prepare()'s auto-wrapping handle everything
output = model(input)                 # autocast + convert_outputs_to_fp32 from prepare()
loss = criterion(output, target)      # fp32 ✓ — no autocast scope, guaranteed fp32
```

Accelerate's `prepare()` handles this automatically: `convert_outputs_to_fp32` wraps
`model.forward` so outputs are always fp32 before they reach your loss function.

**That's why you don't need `accelerator.autocast()` in most training loops** — the model's
forward is already wrapped, and you *want* the loss to be outside autocast scope.

Note: Common loss functions like `nn.CrossEntropyLoss` and `nn.BCELoss` are in autocast's
fp32 promotion list, so they run in fp32 even under autocast. But custom losses with matmuls
or other low-precision ops may silently run in bf16 — this is why keeping loss outside autocast
scope is the safest approach.

### 2. Creating Optimizer Before Upcasting

```python
# WRONG order
optimizer = AdamW(model.parameters())  # optimizer sees bf16 params
model = upcast_to_fp32(model)          # params now fp32, but optimizer
                                       # still has refs to old bf16 tensors!

# CORRECT order
model = upcast_to_fp32(model)          # params now fp32
optimizer = AdamW(model.parameters())  # optimizer sees fp32 params
```

Accelerate's `prepare()` handles this by swapping optimizer param references after FSDP wrapping.

### 3. The `param = param.to()` vs `param.data = param.data.to()` Trap

```python
# WRONG — only rebinds local variable
param = param.to(torch.float32)

# CORRECT — mutates actual parameter
param.data = param.data.to(torch.float32)
```

This was the exact bug we fixed in FSDP2.

### 4. Gradient Accumulation with fp16

```python
# WRONG — accumulating in fp16 loses precision
for micro_batch in micro_batches:
    with accelerator.autocast():
        loss = model(micro_batch) / num_micro_batches
    accelerator.backward(loss)  # grads accumulate in fp16

# BETTER — accumulate in fp32
# (Accelerate handles this internally when you use gradient_accumulation_steps)
with accelerator.accumulate(model):
    loss = model(batch)
    accelerator.backward(loss)
```

### 5. Autocast Weight Cache Memory Overhead

`torch.autocast` caches bf16 copies of fp32 weights so it doesn't re-cast them every forward pass.
This means during forward, both the fp32 master weight and its bf16 cached copy exist in memory:

```
  Without FSDP (single GPU):
    fp32 weight: 4 bytes/param   ← always in memory
    bf16 cache:  2 bytes/param   ← created on first forward, freed after backward
    Peak: 6 bytes/param just for weights (before gradients/optimizer)

  With FSDP:
    Not an issue — FSDP handles the dtype casting via MixedPrecisionPolicy
    during all-gather, so autocast's weight cache is not used for parameters.
```

For very large models on a single GPU, you can disable the cache with
`torch.autocast(..., cache_enabled=False)` at the cost of re-casting every forward pass.

### 6. Saving Checkpoints with Upcasted Params

When params are upcasted to fp32, checkpoints are saved in fp32 (larger). If you want bf16 checkpoints, cast before saving:

```python
# Checkpoint will be fp32 (master weights)
accelerator.save_state()

# To save in bf16 (smaller), you need to cast explicitly
state_dict = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
```

---

## 13. Decision Guide

```
START
  │
  ▼
Do you have Ampere+ GPU (A100, H100, RTX 3090+)?
  │
  ├── YES ──► Use bf16
  │            • No gradient scaling needed
  │            • Same range as fp32
  │            • 2x faster matmuls
  │            │
  │            ▼
  │       Do you have Hopper+ GPU (H100, H200)?
  │            │
  │            ├── YES ──► Consider fp8 (on top of bf16)
  │            │            • Another 2x speedup for matmuls
  │            │            • Needs TransformerEngine or TorchAO
  │            │            • More complex (per-tensor scaling)
  │            │
  │            └── NO ───► Stick with bf16
  │
  └── NO ───► Use fp16
               • Works on all GPUs (Volta, Turing, etc.)
               • Needs GradScaler (Accelerate handles this)
               • Watch for overflow/underflow


Mixed precision + FSDP?
  │
  ├── cpu_ram_efficient_loading = True?
  │     • State dict broadcast + meta model path
  │     • Upcast must happen to BOTH state dict and meta model
  │     • fully_shard sees fp32 → correct master dtype
  │
  └── cpu_ram_efficient_loading = False?
        • Simple: upcast params directly before fully_shard
        • param.data = param.data.to(torch.float32)
```

### Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│  MIXED PRECISION CHEAT SHEET                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Master weights:    ALWAYS fp32                                  │
│  Forward/backward:  bf16/fp16/fp8 (via autocast or FSDP policy) │
│  Optimizer step:    ALWAYS fp32 (params + grads + states)        │
│  Gradient scaling:  fp16 only (GradScaler)                       │
│                                                                  │
│  Loss computation:  ALWAYS fp32 (small values, precision matters)│
│  Autocast:          prepare() auto-wraps model.forward           │
│                     accelerator.autocast() is for non-model code │
│                                                                  │
│  Accelerate API:                                                 │
│    Accelerator(mixed_precision="bf16")  — that's it!            │
│                                                                  │
│  FSDP2 specifics:                                                │
│    fully_shard() snapshots param dtype → upcast BEFORE wrapping  │
│    MixedPrecisionPolicy(param_dtype=bf16) → casts during fwd    │
│    Gradients auto-cast back to master dtype after backward       │
│                                                                  │
│  The golden rule:                                                │
│    param.data = param.data.to(fp32)   ← correct in-place        │
│    param = param.to(fp32)             ← WRONG, local rebind     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
