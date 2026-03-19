# Pure PyTorch Tensor Parallel Tutorial

> **PyTorch version**: 2.6+ required, 2.10+ recommended.
> **Warning**: The Tensor Parallel APIs are marked **experimental** and are subject to change. Always check the [official docs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) for your version.

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [The Big Idea Behind Tensor Parallelism](#2-the-big-idea-behind-tensor-parallelism)
3. [The Two Pieces You Need to Know](#3-the-two-pieces-you-need-to-know)
4. [Setup and Imports](#4-setup-and-imports)
5. [Minimal Example: Sharding a Single Linear Layer](#5-minimal-example-sharding-a-single-linear-layer)
6. [Sharding an MLP](#6-sharding-an-mlp)
7. [Sharding a Transformer Block](#7-sharding-a-transformer-block)
8. [Sequence Parallel](#8-sequence-parallel)
9. [Loss Parallel](#9-loss-parallel)
10. [Debugging with CommDebugMode](#10-debugging-with-commDebugMode)
11. [Combining TP with FSDP2](#11-combining-tp-with-fsdp2)
12. [Running with torchrun](#12-running-with-torchrun)
13. [Common Pitfalls](#13-common-pitfalls)

---

## 1. The Core Problem

You have a giant model and it does not fit on one GPU. What do you do?

The naive answer is: put a copy on every GPU and split the *data* across them. That is Data Parallel training (DDP/FSDP). Each GPU holds the whole model, processes a different batch slice, and the gradients are averaged at the end of each step. It works great until the model itself does not fit in a single GPU's memory, even with batch size 1.

At that point, you need to split the model *itself* across GPUs. Tensor Parallelism is one way to do that.

---

## 2. The Big Idea Behind Tensor Parallelism

Think about a single matrix multiplication: `output = input @ weight`.

Say `weight` is a matrix of shape `(512, 2048)`: 512 input features, 2048 output features. It takes up `512 * 2048 * 4 bytes = 4 MB` in float32. Not huge, but in a real LLM you might have thousands of such matrices, and they can be far larger.

Now imagine you have 4 GPUs. Instead of putting the full `(512, 2048)` weight on every GPU, you split it:

- GPU 0 gets columns 0..511 of the weight: shape `(512, 512)`
- GPU 1 gets columns 512..1023: shape `(512, 512)`
- GPU 2 gets columns 1024..1535: shape `(512, 512)`
- GPU 3 gets columns 1536..2047: shape `(512, 512)`

Each GPU runs its own smaller matmul on the same input and gets a partial output. At the end, you glue the partial results together (by concatenating or summing, depending on the pattern) to get the same answer you would have gotten from the full weight on one GPU.

That is it. That is Tensor Parallelism. Weights are split ("sharded") across GPUs, compute is split accordingly, and you synchronize just enough at the end to produce the correct output.

The "just enough synchronization" part is the engineering challenge. PyTorch's TP library handles it for you automatically based on a sharding plan you specify.

### How does it differ from DDP/FSDP?

| Technique | What is split? | Who holds the full weight at any given moment? |
|-----------|---------------|----------------------------------------------|
| DDP | Data (batch) | Every GPU, always |
| FSDP | Data + weight storage | No one (reconstructed on demand per layer) |
| Tensor Parallel | Weight columns/rows | No one (each GPU holds its shard permanently) |

With DDP and FSDP, each forward pass eventually sees the complete weight. With TP, each forward pass only ever touches a shard. The GPUs cooperate via collective operations (all-reduce, all-gather, reduce-scatter) to produce the correct output.

---

## 3. The Two Pieces You Need to Know

Before writing any code, you need to understand two abstractions: **DTensor** and **DeviceMesh**.

### DTensor: a tensor that knows it is sharded

A `DTensor` is a subclass of `torch.Tensor`. From the outside it looks and feels like a normal tensor. But internally it knows:

- Which GPUs it lives on
- How it is split across those GPUs

When you run an operator on a `DTensor`, PyTorch automatically figures out what collective communication is needed to produce the correct result and fires it for you. You write the math as if everything were on one GPU; the sharding is transparent.

A `DTensor` is described by two things: a **DeviceMesh** (covered below) and a list of **Placements**. There are three placement types:

| Placement | What it means in plain English |
|-----------|-------------------------------|
| `Shard(dim)` | The tensor is cut into equal slices along `dim`. GPU 0 gets slice 0, GPU 1 gets slice 1, and so on. Stack all the slices back along `dim` and you get the original tensor. |
| `Replicate()` | Every GPU has an identical full copy. |
| `Partial()` | Every GPU has a partial value. The actual correct value is what you would get if you summed all the per-GPU values. This shows up transiently during matmuls (before the all-reduce fires) and you rarely need to think about it directly. |

You will mostly deal with `Shard` and `Replicate`. `Partial` is an internal transient state that the library resolves automatically.

### DeviceMesh: a map of your GPUs

`DeviceMesh` is a logical arrangement of your GPU ranks. For basic TP you use a flat 1-D arrangement:

```python
from torch.distributed.device_mesh import init_device_mesh

# "I have 4 GPUs and I want to use them all for tensor parallelism."
tp_mesh = init_device_mesh("cuda", (4,))
```

For combined TP + data parallelism you use a 2-D mesh:

```python
# 16 GPUs: 4-way data parallel, 4-way tensor parallel.
# Think of it as a 4x4 grid of GPUs.
mesh_2d = init_device_mesh("cuda", (4, 4), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh_2d["tp"]  # slice out the TP dimension
dp_mesh = mesh_2d["dp"]  # slice out the DP dimension
```

The mesh tells the TP library which GPUs need to talk to each other during sharded matmuls.

---

## 4. Setup and Imports

Every distributed script needs a process group. `torchrun` spawns one process per GPU and sets the required environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, etc.) before your script starts. Your script just calls `init_process_group`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    loss_parallel,
)


def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def teardown():
    dist.destroy_process_group()
```

---

## 5. Minimal Example: Sharding a Single Linear Layer

Let's do the smallest possible thing: shard one `nn.Linear` across 2 GPUs and run a forward pass.

```python
# minimal_tp.py
# Run: torchrun --standalone --nproc_per_node=2 minimal_tp.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel


class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 16)  # weight shape: (16, 8)

    def forward(self, x):
        return self.fc(x)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # Step 1: Build a 1-D device mesh over 2 GPUs.
    tp_mesh = init_device_mesh("cuda", (2,))

    model = SingleLinear().cuda()

    # Step 2: Tell PyTorch how to shard the model.
    #
    # "fc": ColwiseParallel() means: shard the output dimension of self.fc
    # column-wise across the 2 GPUs.
    #
    # Before TP: weight is (16, 8) on every GPU (replicated).
    # After TP:  GPU 0 holds (8, 8), GPU 1 holds (8, 8).
    #            Together they still represent the full (16, 8) matrix.
    model = parallelize_module(
        model,
        tp_mesh,
        {"fc": ColwiseParallel()},
    )

    # Step 3: Run a forward pass.
    # The input is a plain torch.Tensor -- identical on each rank.
    x = torch.randn(4, 8, device="cuda")  # [batch=4, in_features=8]
    out = model(x)

    # out is a DTensor. Its local shard on this GPU has shape [4, 8]
    # (half of the 16 output features).
    # Logically, across both GPUs, the full output is [4, 16].
    if rank == 0:
        print(f"Local output shape on GPU 0: {out.to_local().shape}")
        # torch.Size([4, 8])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### What just happened, step by step

1. We built a model with a weight of shape `(16, 8)` on each GPU.
2. `parallelize_module` converted `model.fc.weight` from a plain tensor to a `DTensor` with placement `Shard(0)`. The weight's first dimension (16) is now split: GPU 0 holds rows 0..7, GPU 1 holds rows 8..15.
3. On the forward pass, each GPU multiplied its local `[4, 8]` input against its local `[8, 8]` weight shard and got a `[4, 8]` partial output.
4. Because `ColwiseParallel` defaults to keeping the output sharded (`Shard(-1)`), no all-reduce fires here. The outputs stay split across GPUs.

If you wanted the output gathered back into a full `[4, 16]` tensor on every GPU, you would pass `output_layouts=Replicate()` to `ColwiseParallel`. But in practice you do not do that, because the next layer (`RowwiseParallel`) is specifically designed to accept a sharded input directly.

---

## 6. Sharding an MLP

Modern transformer FFN layers follow the SwiGLU pattern:

```
output = W2( silu(W1(x)) * W3(x) )
```

`W1` and `W3` both take the same input `x` and produce independent outputs that get multiplied element-wise. Then `W2` projects back down. This structure is perfect for TP because:

- `W1` and `W3` can be sharded **column-wise**: each GPU computes its chunk of the output independently (both GPUs used the same full input, so no inter-GPU communication is needed after the matmul).
- `W2` is naturally sharded **row-wise**: it accepts the already-sharded outputs of `W1` and `W3`, and its output is combined across GPUs via a single all-reduce.

The whole FFN therefore costs exactly **one all-reduce per forward pass**.

```python
# swiglu_mlp_tp.py
# Run: torchrun --standalone --nproc_per_node=4 swiglu_mlp_tp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


class SwiGLU_MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    tp_mesh = init_device_mesh("cuda", (world_size,))

    DIM = 512
    HIDDEN_DIM = 1024

    model = SwiGLU_MLP(DIM, HIDDEN_DIM).cuda()

    if rank == 0:
        print(f"w1 weight shape BEFORE TP: {model.w1.weight.shape}")
        # (1024, 512)

    tp_plan = {
        # ColwiseParallel: cut w1/w3 along their output dimension.
        # With 4 GPUs: (1024, 512) -> (256, 512) per GPU.
        # Input is assumed replicated (same on all GPUs).
        # Output is sharded on the last dimension (Shard(-1)).
        "w1": ColwiseParallel(),
        "w3": ColwiseParallel(),
        # RowwiseParallel: cut w2 along its input dimension.
        # With 4 GPUs: (512, 1024) -> (512, 256) per GPU.
        # Input is assumed sharded on last dim -- matches w1/w3 output.
        # Output is replicated via an all-reduce.
        "w2": RowwiseParallel(),
    }

    model = parallelize_module(model, tp_mesh, tp_plan)

    if rank == 0:
        print(f"w1 local weight shape AFTER TP: {model.w1.weight.to_local().shape}")
        # (256, 512) -- each GPU holds 256 of the 1024 output columns

    B, S = 2, 32
    x = torch.randn(B, S, DIM, device="cuda")
    out = model(x)

    # out is a plain torch.Tensor because RowwiseParallel all-reduced it.
    if rank == 0:
        print(f"Output shape: {out.shape}")  # [2, 32, 512]

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### The communication flow, drawn out

```
All 4 GPUs receive the same x: [B, S, 512]

W1 forward (ColwiseParallel):
  GPU 0: x @ W1_shard0  ->  [B, S, 256]
  GPU 1: x @ W1_shard1  ->  [B, S, 256]
  GPU 2: x @ W1_shard2  ->  [B, S, 256]
  GPU 3: x @ W1_shard3  ->  [B, S, 256]
  (W3 runs in parallel with the same pattern)

silu(W1_out) * W3_out:
  Happens locally on each GPU, no communication needed.
  Each GPU has its own [B, S, 256] chunk of the intermediate result.

W2 forward (RowwiseParallel):
  GPU 0: [B, S, 256] @ W2_shard0  ->  partial [B, S, 512]
  GPU 1: [B, S, 256] @ W2_shard1  ->  partial [B, S, 512]
  GPU 2: [B, S, 256] @ W2_shard2  ->  partial [B, S, 512]
  GPU 3: [B, S, 256] @ W2_shard3  ->  partial [B, S, 512]

  ---- all-reduce (sum across all 4 GPUs) ----

  All GPUs now have the correct full output: [B, S, 512]
```

---

## 7. Sharding a Transformer Block

A full `TransformerBlock` adds a self-attention layer on top of the FFN. Attention has its own set of linear layers (`wq`, `wk`, `wv`, `wo`) that we also want to shard. The pattern is the same column-wise / row-wise split, but there is one important subtlety covered below.

### 7.1 Model Definition

```python
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape: [B, S, dim] -> [B, n_heads, S, head_dim].
        # When TP is active and use_local_output=False, q/k/v are DTensors.
        # self.n_heads here refers to the LOCAL n_heads (total / tp_size).
        # DTensor is aware of the sharding so the view works correctly.
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        out = torch.matmul(scores, v)  # [B, n_heads, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = FeedForward(dim, ffn_hidden_dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

### 7.2 The sneaky problem: view operations after a sharded matmul

This is the most confusing part of the whole API, so let's be very explicit.

After sharding `wq` column-wise with 4-way TP, its output has local shape `[B, S, dim // 4]` on each GPU. Then in `Attention.forward` we do:

```python
q = self.wq(x)
q = q.view(B, S, self.n_heads, self.head_dim)
```

`self.n_heads` is the *global* number of heads (say, 8). But the local tensor only has `dim // 4` features in the last dimension, which corresponds to only 2 local heads. Calling `.view(B, S, 8, head_dim)` on the local tensor would fail or silently produce garbage because `8 * head_dim != dim // 4`.

The fix is `use_local_output=False`. This tells `ColwiseParallel` to keep the output as a `DTensor` rather than unwrapping it to a plain `torch.Tensor`. A `DTensor` knows it is sharded, so when you call `.view(B, S, n_heads, head_dim)` on it, it understands that `n_heads` refers to the *local* number of heads. Everything works correctly.

**Rule of thumb**: use `use_local_output=False` whenever a view or reshape follows the column-wise projection.

### 7.3 The TP Plan

```python
def build_tp_plan():
    return {
        # wq/wk/wv: column-wise, split by heads.
        # use_local_output=False: keep as DTensor so the subsequent
        # view(B, S, n_heads, head_dim) works on the local head count.
        "attention.wq": ColwiseParallel(use_local_output=False),
        "attention.wk": ColwiseParallel(use_local_output=False),
        "attention.wv": ColwiseParallel(use_local_output=False),
        # wo: row-wise, all-reduces the partial attention output.
        "attention.wo": RowwiseParallel(),
        # FFN: identical to the MLP example.
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
    }
```

### 7.4 Full Runnable Script

```python
# transformer_tp.py
# Run: torchrun --standalone --nproc_per_node=4 transformer_tp.py

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    DIM = 256
    N_HEADS = 8        # must be divisible by world_size
    N_LAYERS = 2
    FFN_HIDDEN = 512

    assert N_HEADS % world_size == 0, (
        f"n_heads ({N_HEADS}) must be divisible by tp_size ({world_size})"
    )

    tp_mesh = init_device_mesh("cuda", (world_size,))
    tp_plan = build_tp_plan()

    layers = nn.ModuleList([
        TransformerBlock(DIM, N_HEADS, FFN_HIDDEN) for _ in range(N_LAYERS)
    ]).cuda()

    for block in layers:
        parallelize_module(block, tp_mesh, tp_plan)

    B, S = 2, 16
    x = torch.randn(B, S, DIM, device="cuda", requires_grad=True)
    out = x
    for block in layers:
        out = block(out)

    loss = out.sum()
    loss.backward()

    if rank == 0:
        print(f"Output shape: {out.shape}")      # [2, 16, 256]
        print(f"Input grad shape: {x.grad.shape}")  # [2, 16, 256]

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

## 8. Sequence Parallel

### The problem it solves

In basic TP, the activations *between* the attention and FFN sub-layers are **replicated** on every GPU. With 4-way TP and a `[B, S, D]` activation, each GPU holds the entire `[B, S, D]` tensor even though there are 4 of them doing the same redundant storage. That wastes memory.

Sequence Parallel (SP) fixes this by keeping those "between-layer" activations **sharded along the sequence dimension** instead. With 4-way TP+SP, each GPU stores `[B, S/4, D]`, cutting activation memory by 4x.

### How it works

Norm layers like `RMSNorm` and `LayerNorm` operate independently on each token. It does not matter if each GPU only sees `S/4` tokens -- the output is the same as if it saw the full `S` tokens and then took its slice. Their *parameters* are replicated; the *computation* runs on a sequence shard.

The flow per transformer block becomes:

```
Between blocks: activation is [B, S/4, D]  (sequence-sharded, cheap to store)

  RMSNorm (SequenceParallel)
    -> runs locally on [B, S/4, D], no communication needed

  all-gather at attention boundary
    -> [B, S/4, D] x 4 GPUs  ==>  [B, S, D] replicated on all GPUs

  Attention + FFN (same as basic TP)
    -> runs with replicated input, sharded weights

  reduce-scatter at output
    -> [B, S, D] partial  ==>  [B, S/4, D] on each GPU

Back between blocks: activation is [B, S/4, D] again
```

The all-gather + reduce-scatter pair replaces the single all-reduce from basic TP. Total communication volume is identical, but the big activation tensors between blocks are now sequence-sharded, saving memory.

### The Code

```python
def build_sp_plan():
    """TP + Sequence Parallel plan."""
    return {
        # Norm layers: parameters replicated, but computation runs on S/4 tokens.
        "attention_norm": SequenceParallel(),

        # At the attention boundary: all-gather from Shard(1) -> Replicate().
        # Shard(1) means sharded on dim=1, which is the sequence dimension [B, S, D].
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        # Same attention sharding as before.
        "attention.wq": ColwiseParallel(use_local_output=False),
        "attention.wk": ColwiseParallel(use_local_output=False),
        "attention.wv": ColwiseParallel(use_local_output=False),
        # wo: instead of all-reduce (Partial -> Replicate),
        # we reduce-scatter (Partial -> Shard(1)).
        # This produces sequence-sharded output for the next norm layer.
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),

        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
        # Same reduce-scatter trick for w2.
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    }
```

For SP to chain correctly across layers, the embedding and final projection also need explicit layout annotations so the sequence-sharded "contract" is maintained end to end:

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),   # produce sequence-sharded embeddings
        ),
        "norm": SequenceParallel(),    # final norm before output projection
        "output": ColwiseParallel(
            input_layouts=Shard(1),    # accept sequence-sharded input
            output_layouts=Replicate(),
        ),
    },
)
```

---

## 9. Loss Parallel

### The problem it solves

The final output projection in a language model maps from hidden dim to vocabulary: `nn.Linear(dim, vocab_size)`. Vocabulary sizes are large (32k to 128k). If `vocab_size=128000` and `dim=4096`, this single matrix is about 1 GB in bfloat16.

After column-wise sharding of this projection, each GPU only holds `vocab_size // tp_size` output columns. But to compute cross-entropy loss you normally need the full logit vector for every token. Gathering those logits to every GPU uses a lot of memory and bandwidth.

`loss_parallel` computes cross-entropy loss *without gathering the full logits*. It splits the log-sum-exp computation across GPUs: each GPU computes the local max and local sum for its vocab shard, a tiny all-reduce exchanges two scalars, and each GPU computes its contribution to the final loss. The math is numerically identical to computing loss on the gathered logits, with far less memory and communication.

### How to use it

Two requirements:

1. The output projection must use `ColwiseParallel` with `use_local_output=False` so the logits stay as a `DTensor` (sharded on the vocab dimension).
2. Both the loss computation and `loss.backward()` must happen inside `with loss_parallel():`.

```python
# Apply to the output head.
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            use_local_output=False,  # keep as DTensor for loss_parallel
        ),
    },
)

# In your training loop:
pred = model(input_ids)  # DTensor, vocab dim sharded: [B, S, vocab // tp_size]

with loss_parallel():
    loss = F.cross_entropy(
        pred.flatten(0, 1),     # [B*S, vocab] as a DTensor
        labels.flatten(0, 1),   # [B*S] as a plain Tensor
    )
    loss.backward()  # backward must also be inside the context
```

---

## 10. Debugging with CommDebugMode

When you first set up a TP plan it is easy to make a mistake: a wrong FQN, a missing entry, or a wrong layout. `CommDebugMode` shows you exactly what collectives fired during a forward/backward pass so you can verify your plan is doing what you think.

```python
from torch.distributed.tensor.debug import CommDebugMode

comm_mode = CommDebugMode()

with comm_mode:
    out = model(x)
    out.sum().backward()

# High-level count of each collective type.
print(comm_mode.get_comm_counts())
# Example output for a 2-layer model with basic TP (no SP):
# {<class 'torch.ops.c10d_functional.all_reduce'>: 4}
# 2 layers * (1 attention all-reduce + 1 FFN all-reduce) = 4. Correct.

# Module-level breakdown. noise_level: 0=module counts, 1=DTensor ops, 2=everything.
print(comm_mode.generate_comm_debug_tracing_table(noise_level=1))
```

With Sequence Parallel you should see zero `all_reduce` operations and instead see `all_gather` and `reduce_scatter` at the module boundaries. An unexpected `all_reduce` inside SP mode means a layout conversion is happening that you did not intend.

---

## 11. Combining TP with FSDP2

TP shards weights across GPUs *within* a node (they talk via fast NVLink). FSDP2 shards weights *across* nodes (they talk via slower inter-node networking). These two dimensions compose naturally into a 2-D parallelism strategy.

The rule is: **apply TP first, then FSDP2.** After `parallelize_module`, the weights are `DTensor`s sharded across the TP mesh. Then `fully_shard` wraps the model and shards those `DTensor` parameters further across the data-parallel dimension.

```python
# fsdp2_tp.py
# Run: torchrun --standalone --nproc_per_node=8 fsdp2_tp.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.fsdp import fully_shard  # FSDP2


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # 8 GPUs: 2-way DP x 4-way TP.
    tp_size = 4
    dp_size = world_size // tp_size  # 8 // 4 = 2

    mesh_2d = init_device_mesh(
        "cuda",
        (dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    tp_mesh = mesh_2d["tp"]
    dp_mesh = mesh_2d["dp"]

    class TwoLayerMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, 512, bias=False),
                    nn.ReLU(),
                    nn.Linear(512, 256, bias=False),
                )
                for _ in range(2)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = TwoLayerMLP().cuda()

    # Step 1: Apply TP (intra-node, via NVLink).
    for seq_layer in model.layers:
        parallelize_module(
            module=seq_layer,
            device_mesh=tp_mesh,
            parallelize_plan={
                "0": ColwiseParallel(),  # first Linear (index 0 in Sequential)
                "2": RowwiseParallel(),  # second Linear (index 2 in Sequential)
            },
        )

    # Step 2: Apply FSDP2 on top (inter-node).
    # FSDP2 sees the TP-sharded DTensor parameters and shards them
    # further across dp_mesh.
    model = fully_shard(model, mesh=dp_mesh)

    # Step 3: Train normally.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(4, 256, device="cuda")
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if rank == 0:
        print("TP + FSDP2 step OK.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

## 12. Running with torchrun

All scripts must be launched with `torchrun`, not with `python` directly. `torchrun` handles spawning one process per GPU, assigning ranks, and setting up the rendezvous.

**Single node, N GPUs:**

```bash
torchrun --standalone --nproc_per_node=N script.py
```

**Multi-node (e.g. 2 nodes, 4 GPUs each, world_size=8):**

```bash
# Run this same command on both nodes.
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<node0_ip>:29500 \
  script.py
```

Do not use `torch.multiprocessing.spawn` for TP workloads. `torchrun` is the correct launcher.

---

## 13. Common Pitfalls

### n_heads must divide evenly by tp_size

TP shards attention by heads. If you have 8 heads and 3-way TP, you cannot split 8 heads into 3 equal groups. This will produce wrong results or crash at the reshape step.

```python
assert n_heads % tp_size == 0, (
    f"n_heads ({n_heads}) must be divisible by tp_size ({tp_size})"
)
```

### parallelize_module only accepts a 1-D DeviceMesh

If you have a 2-D mesh for TP + DP, you must slice it before passing it to `parallelize_module`.

```python
mesh_2d = init_device_mesh("cuda", (4, 4), mesh_dim_names=("dp", "tp"))

# WRONG: will raise an error.
parallelize_module(model, mesh_2d, plan)

# CORRECT:
parallelize_module(model, mesh_2d["tp"], plan)
```

### Wrong FQN keys silently do nothing

`parallelize_module` does not raise an error if a key in your plan does not match any submodule. It just skips it. If you typo a layer name, the layer will not be sharded and you will get no warning. Always verify with `CommDebugMode`.

```python
# "feed_foward" has a typo (one 'r'). This is silently ignored.
# The FFN will NOT be sharded. Training will appear to work but use too much memory.
plan = {"feed_foward.w1": ColwiseParallel(), ...}
```

### All TP ranks must receive the same input

TP computes a sharded matmul where every GPU uses the *same* input with a different weight shard. If the inputs differ across ranks, the all-reduce will produce the wrong answer and training will diverge without any error message. Make sure your data loader does not shuffle inputs independently per rank within a TP group.

### loss.backward() must be inside `with loss_parallel()`

```python
# WRONG: backward is outside the context.
with loss_parallel():
    loss = F.cross_entropy(pred.flatten(0,1), labels.flatten(0,1))
loss.backward()

# CORRECT:
with loss_parallel():
    loss = F.cross_entropy(pred.flatten(0,1), labels.flatten(0,1))
    loss.backward()
```

### ColwiseParallel unwraps output to a plain tensor by default

`ColwiseParallel` has `use_local_output=True` by default, meaning the output is unwrapped from `DTensor` to a plain `torch.Tensor`. This is fine for FFN layers but wrong for attention projections where a reshape follows. Always use `use_local_output=False` for `wq`, `wk`, `wv`.

---

## Summary: The Full API Surface

```python
# 1. Device topology
from torch.distributed.device_mesh import init_device_mesh

# 2. Placement types
from torch.distributed.tensor import Shard, Replicate, Partial

# 3. The sharding plan primitives
from torch.distributed.tensor.parallel import (
    parallelize_module,    # entrypoint: apply a plan to an nn.Module
    ColwiseParallel,       # shard nn.Linear/Embedding on the output (column) dim
    RowwiseParallel,       # shard nn.Linear on the input (row) dim
    SequenceParallel,      # run norm layers with sequence-sharded activations
    PrepareModuleInput,    # insert a collective at a module's input boundary
    PrepareModuleOutput,   # insert a collective at a module's output boundary
    loss_parallel,         # context manager for sharded cross-entropy
)

# 4. Debugging
from torch.distributed.tensor.debug import CommDebugMode
```
