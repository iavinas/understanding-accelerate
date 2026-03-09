# Level 2: Multi-GPU Training with DistributedDataParallel

> **Series context.** This chapter assumes you have completed Level 1 and are
> comfortable with the five-line Accelerate integration pattern, the
> `PartialState` / `AcceleratorState` Borg singleton, `DistributedType`, and
> the three branches of `backward()`. Here we move from a single-device world
> into one where multiple processes run simultaneously and must coordinate.
> Every concept introduced below is grounded in the actual source files; file
> paths and structural landmarks are called out explicitly so you can verify
> every claim against the repository.

---

## 1. The Mental Model: Parallel Processes, Not Parallel Threads

Before touching any code, it is worth being precise about what
DistributedDataParallel (DDP) actually does, because many training bugs stem
from a fuzzy mental model.

PyTorch DDP is **process-level** parallelism, not thread-level parallelism. When
you run a DDP job across two GPUs, you get two completely independent Python
interpreter processes. Each process loads the full model into its own GPU
memory. Each process runs the full forward pass on its own shard of data.
After the backward pass, the processes communicate to average the gradients
and then each independently applies the same optimizer step. Because the
starting weights and the averaged gradients are identical on every process, the
weights remain in sync without any further coordination.

The implication is significant: **any code that does not touch the gradient
averaging step runs independently on every process.** If you call `print()` in
your training loop, you will see it printed once per process. If you write a
checkpoint file without guarding on `process_index`, every process will try to
write to the same path simultaneously.

Accelerate's job is to hide almost all of this complexity while still giving
you the escape hatches you need when the abstraction leaks.

---

## 2. `accelerate launch`: What Actually Happens

### 2.1 The CLI entry point

The `accelerate launch` command is defined in
`src/accelerate/commands/launch.py`. The file parses a rich set of CLI
arguments, then delegates to one of several launcher back-ends depending on the
target platform (NCCL multi-GPU, DeepSpeed, FSDP, TPU, etc.).

For ordinary multi-GPU DDP on a single machine the relevant branch is the
`multi_gpu_launcher`, which ultimately calls `torch.distributed.run` (the same
binary that backs the `torchrun` command). The rough call chain is:

```
accelerate launch --num_processes=2 train.py
   └─ launch_command()                    # launch.py
       └─ multi_gpu_launcher(args, ...)   # launch.py
           └─ torch.distributed.run(...)  # delegates to torchrun internally
               ├─ spawns process 0  → runs train.py with LOCAL_RANK=0
               └─ spawns process 1  → runs train.py with LOCAL_RANK=1
```

The key insight is that `accelerate launch` is fundamentally a thin wrapper
around PyTorch's own distributed launcher. Its value-add is (a) reading a YAML
config file so you do not have to remember every flag, and (b) abstracting over
multiple back-ends so the same command works for DeepSpeed and TPUs without
code changes.

### 2.2 Environment variables

When `torch.distributed.run` spawns your processes it sets a standard set of
environment variables before each process's Python interpreter starts:

| Variable | Meaning |
|---|---|
| `WORLD_SIZE` | Total number of processes across all machines |
| `RANK` | Global rank of this process (0 to `WORLD_SIZE - 1`) |
| `LOCAL_RANK` | Rank of this process on the *current machine* |
| `MASTER_ADDR` | Hostname or IP of rank-0 process (rendezvous point) |
| `MASTER_PORT` | TCP port used for the initial rendezvous |

For a single-machine two-GPU job these collapse to `WORLD_SIZE=2`,
`RANK=0`/`1`, `LOCAL_RANK=0`/`1`, and `MASTER_ADDR=localhost`.

`PartialState.__init__()` (in `src/accelerate/state.py`) reads exactly these
variables to populate its fields. That is why `PartialState()` works correctly
whether you call it inside a `torch.distributed.run`-spawned process or a
plain `python train.py` — in the latter case the variables are absent and
Accelerate falls back to `DistributedType.NO`.

### 2.3 The `accelerate config` YAML

Running `accelerate config` interactively produces a YAML file such as:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 2
machine_rank: 0
main_process_ip: null
main_process_port: null
mixed_precision: fp16
fsdp_config: {}
deepspeed_config: {}
use_cpu: false
```

`accelerate launch` reads this file and translates it into the flags that
`torchrun` understands. You can override any field on the command line:

```bash
# Use the YAML for everything except num_processes
accelerate launch --num_processes=4 train.py

# Point to a custom YAML
accelerate launch --config_file configs/a100x8.yaml train.py
```

The YAML is optional. If you omit `accelerate config` and call
`accelerate launch --num_processes=2 train.py` directly, Accelerate infers
reasonable defaults (all available GPUs, no mixed precision).

---

## 3. From Single-GPU to Multi-GPU: The Five-Line Pattern Revisited

Here is the single-GPU training script from Level 1:

```python
# train_single.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

accelerator = Accelerator()

X = torch.randn(1024, 16)
y = torch.randint(0, 2, (1024,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = torch.nn.Linear(16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for epoch in range(3):
    for batch in loader:
        inputs, labels = batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

accelerator.print("Training complete.")
```

To run this on two GPUs you do not change a single line of Python. You only
change how you invoke it:

```bash
# Before (single GPU)
python train_single.py

# After (two GPUs)
accelerate launch --num_processes=2 train_single.py
```

Everything else is handled by the prepare / backward / print
machinery you already know. This is the promise of Accelerate: the script is
device-agnostic by construction.

---

## 4. Data Distribution: Two Modes Compared

When `prepare()` receives a `DataLoader` it cannot simply hand it back
unchanged. The reason is architectural: PyTorch does not allow you to mutate a
DataLoader's `batch_sampler` after construction, so Accelerate must **rebuild**
the DataLoader from scratch, wrapping either the sampler or the loader itself
depending on the mode selected.

The decision is controlled by the `dispatch_batches` field on
`DataLoaderConfiguration` (or its corresponding argument to
`prepare_data_loader()` in `src/accelerate/data_loader.py`).

### 4.1 Sampler mode (`dispatch_batches=False`)

In sampler mode each process gets its own independent DataLoader, but the
underlying `BatchSampler` is replaced with a `BatchSamplerShard`. This
custom sampler yields only the slice of indices that belongs to the current
process:

```
Dataset indices: [0, 1, 2, 3, 4, 5, 6, 7]

GPU 0's BatchSamplerShard sees:  [0, 2, 4, 6]   (even indices)
GPU 1's BatchSamplerShard sees:  [1, 3, 5, 7]   (odd indices)
```

Each process fetches its own data directly from storage (or from its own copy
of the dataset in memory). There is no inter-process communication during data
loading. The result is wrapped in a `DataLoaderShard`, which adds two
additional responsibilities:

1. **RNG synchronization.** At the start of each iteration it synchronizes the
   random number generators across all processes so that shuffling yields the
   same permutation everywhere. (Without this, each process would shuffle
   independently and the global batch would not represent a random sample of
   the full dataset.)
2. **Device placement.** If `device_placement=True`, the shard automatically
   moves each batch to the correct CUDA device before yielding it.

The relevant class in `data_loader.py` is `DataLoaderShard`, a `DataLoader`
subclass whose `__iter__` performs the RNG sync and device placement before
delegating to the parent iterator.

### 4.2 Dispatch mode (`dispatch_batches=True`)

In dispatch mode a single process (rank 0) is responsible for building each
batch and then scattering the pieces to all other processes over the process
group:

```
Dataset: [A, B, C, D, E, F, G, H]   (rank 0 reads the whole batch)

Full batch assembled on rank 0:  [A, B, C, D]
  → rank 0 keeps:  [A, B]
  → rank 1 receives: [C, D]  (via torch.distributed.scatter)
```

This is implemented in `DataLoaderDispatcher`, which subclasses
`DataLoaderAdapter` (itself a thin shim around `DataLoader`) and overrides
`__iter__` to drive the scatter logic. Because all data flows through rank 0,
this mode is slower for large datasets but is the safer default when the
dataset is not easily shardable (for example, with variable-length sequences
where you want rank 0 to handle padding for the entire batch before dispatching
shards).

The `DataLoaderDispatcher` docstring explicitly notes that it "differs from
`DataLoaderShard` in that when iterating through the DataLoader, the data is
all starting from process 0 and then split and sent off to each process rather
than it happening at the dataset level."

### 4.3 Choosing between the two modes

In practice you rarely set `dispatch_batches` explicitly. Accelerate's default
heuristic in `prepare_data_loader()` is:

- If the dataset is an `IterableDataset`, default to `dispatch_batches=True`
  (because `IterableDataset` has no concept of random access indices).
- For map-style datasets, default to `dispatch_batches=False` (sampler mode).

You can override via `DataLoaderConfiguration`:

```python
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

# Force sampler mode even for IterableDataset
dl_config = DataLoaderConfiguration(dispatch_batches=False)
accelerator = Accelerator(dataloader_config=dl_config)
```

### 4.4 The `even_batches` edge case

A subtlety that trips up contributors: what happens when the dataset size is
not evenly divisible by the number of processes?

By default `even_batches=True`. Accelerate pads the last batch by repeating
samples from the beginning of the dataset so that every process always receives
exactly the same number of items. This avoids hangs that would occur if one
process ran out of data while another was still waiting for an all-reduce.

The flip side is that you may see slightly duplicate data in the final batch of
each epoch. For evaluation this matters: `gather_for_metrics()` (discussed in
Section 7) exists precisely to strip those padding samples back out after
gathering.

For training the effect is usually negligible, but if you want strict dataset
iteration without padding, you can disable it:

```python
dl_config = DataLoaderConfiguration(even_batches=False)
accelerator = Accelerator(dataloader_config=dl_config)
```

With `even_batches=False` you must ensure your dataset is divisible by
`num_processes` yourself, or use `drop_last=True` on the original DataLoader.

---

## 5. What `prepare()` Does to the Model

When `prepare()` receives a `torch.nn.Module` under `DistributedType.MULTI_GPU`
it wraps it in `torch.nn.parallel.DistributedDataParallel`:

```python
# Conceptually inside accelerator.py
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    device_ids=[accelerator.local_process_index],
    output_device=accelerator.local_process_index,
    **ddp_kwargs,   # from DistributedDataParallelKwargs handler
)
```

The DDP wrapper installs hooks into the model's backward pass. Every time a
gradient is computed for a parameter, the hook schedules an **all-reduce
operation** that averages that gradient across all processes. By the time
`loss.backward()` returns, every process has the same averaged gradient for
every parameter.

This is both DDP's power and its performance trap. The all-reduce is
**synchronous and blocking**. If you run `backward()` on every mini-batch, you
pay the communication overhead on every mini-batch. Gradient accumulation
(Section 6) is the tool for amortizing that cost.

### 5.1 The `process_group` argument

By default the DDP wrapper uses the global default process group, which
`torch.distributed.init_process_group()` creates automatically when Accelerate
initializes state. For most users this is correct. Advanced users who want to
limit gradient averaging to a subset of processes (for example, model-parallel
sub-groups) can pass a custom process group via `DistributedDataParallelKwargs`.

### 5.2 Accessing the unwrapped model

After `prepare()` the model you hold is a `DDP` object, not your original
module. If you need to call module-specific methods (like saving a
`state_dict`, or accessing custom attributes), you must unwrap it:

```python
unwrapped = accelerator.unwrap_model(model)
# Now unwrapped is your original torch.nn.Module
torch.save(unwrapped.state_dict(), "checkpoint.pt")
```

`unwrap_model()` is a no-op when not in distributed mode, so it is safe to call
unconditionally.

---

## 6. Gradient Accumulation: The Single Biggest Performance Lever

### 6.1 Why you pay an unnecessary communication tax without it

Imagine you want an effective batch size of 128 but can only fit 32 samples
per GPU. The naive approach runs `backward()` every step, triggering an
all-reduce every step. That is four times more communication than necessary,
because you are communicating after each of four micro-batches that will
eventually combine into one optimizer update.

The correct approach is to accumulate gradients locally for four steps, then
communicate once:

```
Step 1: forward(batch_0) → backward → gradients accumulate locally (NO all-reduce)
Step 2: forward(batch_1) → backward → gradients accumulate locally (NO all-reduce)
Step 3: forward(batch_2) → backward → gradients accumulate locally (NO all-reduce)
Step 4: forward(batch_3) → backward → ALL-REDUCE → optimizer.step()
```

This requires suppressing the DDP gradient hook for steps 1-3. PyTorch
provides exactly this facility through the `model.no_sync()` context manager.
Inside `no_sync()`, the backward hooks are disabled, so gradients accumulate
in the parameter's `.grad` tensor locally without triggering any communication.

### 6.2 The naive implementation (do not use this)

To understand what Accelerate automates, here is the manual version without
Accelerate:

```python
accumulation_steps = 4
ddp_model.train()

for step, batch in enumerate(dataloader):
    inputs, labels = batch
    is_last_step = (step + 1) % accumulation_steps == 0

    if not is_last_step:
        # Suppress the all-reduce hook
        with ddp_model.no_sync():
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()  # gradients accumulate locally
    else:
        # Allow the all-reduce to happen
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()     # triggers all-reduce at end of backward
        optimizer.step()
        optimizer.zero_grad()
```

This is tedious and error-prone. You must remember to divide the loss by
`accumulation_steps`, manually track which step is the last, and remember to
call `zero_grad()` only after the optimizer step. Any mistake silently produces
wrong gradients.

### 6.3 The Accelerate implementation

Accelerate automates all of this with two pieces working together: the
`Accelerator(gradient_accumulation_steps=N)` constructor argument and the
`accumulate()` context manager.

```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=4)
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for batch in loader:
    with accelerator.accumulate(model):
        inputs, labels = batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

You write the inner loop as if it runs every step. Accelerate intercepts at
three points:

1. **`accumulate()` entry.** The context manager calls `_do_sync()`, which
   consults `GradientState` to decide whether this is a synchronization step.
   If it is not, the context manager enters `no_sync(model)`, suppressing the
   all-reduce.

2. **`backward()`.** The loss is automatically divided by
   `gradient_accumulation_steps` before calling the underlying
   `loss.backward()`. The division is handled inside
   `GradientState`'s interaction with the backward call. The exact location is
   in `accelerator.py`'s `backward()` method, which checks
   `self.gradient_state.sync_gradients` to decide whether to scale.

3. **`optimizer.step()` and `zero_grad()`.** The wrapped
   `AcceleratedOptimizer.step()` checks `self.gradient_state.sync_gradients`
   and becomes a no-op on non-synchronization steps. `zero_grad()` likewise
   no-ops unless it is time to actually reset.

The result is that you can write the training loop once, idiomatically, and
Accelerate translates it into the correct DDP + gradient accumulation pattern
automatically.

### 6.4 Source tour: `accumulate()` in `accelerator.py`

The actual implementation of `accumulate()` (at the time of writing) is
approximately:

```python
@contextmanager
def accumulate(self, *models):
    self._do_sync()
    with contextlib.ExitStack() as cm_stack:
        for m in models:
            cm_stack.enter_context(
                contextlib.nullcontext()
                if self.sync_gradients
                else self.no_sync(m)
            )
        yield
```

`_do_sync()` is what updates `GradientState.sync_gradients`. It consults the
active dataloader's iteration state (via `GradientState`'s
`active_dataloader` reference) to decide if we have reached the last step in
an accumulation window. The condition is: we are on a step that is a multiple
of `gradient_accumulation_steps`, or we are at the end of the dataloader
(whichever comes first).

`sync_gradients` is then read by `self.no_sync(m)`. If `sync_gradients=True`
the model runs with normal DDP hooks (gradients will be communicated). If
`sync_gradients=False`, the model enters `no_sync()` and gradients stay local.

### 6.5 `GradientState` and `sync_with_dataloader`

`GradientState` is itself a Borg-pattern singleton (same architecture as
`PartialState`). Its key fields are:

| Field | Meaning |
|---|---|
| `num_steps` | The configured `gradient_accumulation_steps` |
| `sync_gradients` | Boolean: should this step do the all-reduce? |
| `active_dataloader` | A weak reference to the current `DataLoaderShard` |
| `end_of_dataloader` | Set to `True` when the dataloader is exhausted |

The `sync_with_dataloader=True` default means: if we reach the end of the
dataloader mid-accumulation-window, force a sync anyway. This prevents a common
bug where the last few batches of an epoch are never used in an optimizer step.
You can disable this behavior with `GradientAccumulationPlugin`:

```python
from accelerate.utils import GradientAccumulationPlugin

plugin = GradientAccumulationPlugin(sync_with_dataloader=False)
accelerator = Accelerator(gradient_accumulation_plugin=plugin)
```

Use this with care: disabling it means that if your dataset is not evenly
divisible by `gradient_accumulation_steps`, the leftover batches at the end of
each epoch will never contribute to a weight update.

### 6.6 FSDP warning: `no_sync` costs memory

For DDP, `no_sync()` is purely a communication optimization with no memory
overhead. For FSDP (Fully Sharded Data Parallel, the subject of Level 4),
`no_sync()` forces FSDP to keep full un-sharded gradient tensors in memory for
the entire accumulation window. On very large models this can cause OOM errors.
If you encounter this, set `sync_each_batch=True` in `GradientAccumulationPlugin`
to disable `no_sync` at the cost of communicating gradients every step:

```python
from accelerate.utils import GradientAccumulationPlugin

plugin = GradientAccumulationPlugin(sync_each_batch=True)
accelerator = Accelerator(gradient_accumulation_plugin=plugin)
```

This issue does not arise in standard DDP training, but it is worth knowing
about because the code path is shared.

---

## 7. Gathering Results for Evaluation

Because each process sees a different shard of the evaluation dataset, you
cannot compute metrics locally per-process and expect them to be correct.
You must gather the predictions and labels from all processes before computing
any metric.

### 7.1 `gather()`

```python
predictions = model(batch_inputs)             # shape: [local_batch, num_classes]
all_predictions = accelerator.gather(predictions)  # shape: [world_size * local_batch, num_classes]
```

`gather()` is a thin wrapper around `torch.distributed.all_gather()`. Every
process contributes its local tensor and every process receives the
concatenated result. The concatenation is along dimension 0.

Because every process gets the full result, any process can compute the final
metric. The convention in Accelerate examples is to compute metrics only on
rank 0 (`accelerator.is_main_process`) to avoid printing the same value
`num_processes` times.

### 7.2 `gather_for_metrics()`: handling uneven batches

`gather()` has a correctness problem during evaluation when the dataset size is
not evenly divisible by `num_processes`. The DataLoader will have padded the
final batch with duplicate samples (because `even_batches=True` by default).
Those duplicate samples will be gathered along with the real ones, inflating
your metric.

`gather_for_metrics()` solves this by tracking how many real samples each
process contributed and stripping the padding samples after gathering:

```python
all_predictions = accelerator.gather_for_metrics(predictions)
all_labels      = accelerator.gather_for_metrics(labels)

# Only on rank 0, compute the metric
if accelerator.is_main_process:
    accuracy = (all_predictions.argmax(dim=1) == all_labels).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
```

`gather_for_metrics()` is the correct default for any evaluation metric.
`gather()` should only be used when you know your dataset is perfectly
divisible or when you intentionally want the padded samples.

### 7.3 `gather_object()`: non-tensor data

Both `gather()` and `gather_for_metrics()` require tensors. For arbitrary
Python objects (strings, dicts, lists of variable-length sequences) use
`gather_object()`, which internally calls
`torch.distributed.all_gather_object()`:

```python
local_results = [{"text": "...", "score": 0.91}]
all_results = accelerator.gather_object(local_results)
# all_results is a flat list with contributions from all processes
```

Be aware that `all_gather_object()` uses pickle serialization and is
significantly slower than tensor all-gather for large payloads. For
performance-sensitive evaluation loops, prefer to gather tensors and
reconstruct objects afterward.

### 7.4 Process-aware printing

A subtle point: if you call `print()` inside the evaluation loop, all processes
will print. Use `accelerator.print()` to restrict output to rank 0:

```python
accelerator.print(f"Epoch {epoch}: accuracy={accuracy:.4f}")
# Identical to: if accelerator.is_main_process: print(...)
```

To observe that processes really are running in isolation, you can deliberately
use all-process printing:

```python
print(f"[rank {accelerator.process_index}] local accuracy = {local_acc:.4f}")
```

---

## 8. Complete Training Script: Putting It All Together

The following script is a self-contained, runnable example that demonstrates
every concept from this chapter: multi-process launch, two-mode data loading,
gradient accumulation, and correct evaluation with `gather_for_metrics()`.

```python
# train_ddp.py
"""
Run with:
    accelerate launch --num_processes=2 train_ddp.py

Or with a config file:
    accelerate launch --config_file my_config.yaml train_ddp.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from accelerate import Accelerator

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE          = 32
GRADIENT_ACCUM      = 4     # effective batch = 32 * 4 = 128 per GPU
EPOCHS              = 5
LR                  = 1e-3
HIDDEN              = 64
INPUT_DIM           = 32
NUM_CLASSES         = 4
N_TRAIN             = 2048
N_EVAL              = 512

# ── Model ────────────────────────────────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Initialize Accelerator with gradient accumulation
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUM)

    # Every process reports its identity. This demonstrates process isolation.
    print(
        f"[rank {accelerator.process_index}/{accelerator.num_processes}] "
        f"device={accelerator.device}"
    )

    # 2. Build dataset (identical on every process before prepare())
    torch.manual_seed(42)
    X = torch.randn(N_TRAIN + N_EVAL, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (N_TRAIN + N_EVAL,))
    train_ds, eval_ds = random_split(
        TensorDataset(X, y), [N_TRAIN, N_EVAL]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # 3. Build model, optimizer, scheduler
    model     = SmallMLP(INPUT_DIM, HIDDEN, NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    # 4. prepare() — this is where DDP wrapping and DataLoader rebuilding happen
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # 5. Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_updates = 0

        for batch in train_loader:
            with accelerator.accumulate(model):
                inputs, labels = batch
                outputs  = model(inputs)
                loss     = criterion(outputs, labels)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Only count losses on sync steps to get a meaningful average
            if accelerator.sync_gradients:
                total_loss  += loss.detach()
                num_updates += 1

        # 6. Evaluation with gather_for_metrics
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                inputs, labels = batch
                outputs  = model(inputs)
                preds    = outputs.argmax(dim=-1)

                # Gather from all processes; strips padding automatically
                preds  = accelerator.gather_for_metrics(preds)
                labels = accelerator.gather_for_metrics(labels)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy   = (all_preds == all_labels).float().mean()

        # 7. Print only once across all processes
        accelerator.print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"loss={total_loss/num_updates:.4f} | "
            f"eval_acc={accuracy:.4f}"
        )

    accelerator.print("Training complete.")

if __name__ == "__main__":
    main()
```

A few details worth noting in this script:

The dataset construction happens identically on every process before
`prepare()` is called. This is fine because `prepare()` will then arrange for
each process to see a different shard via its `BatchSamplerShard`. If dataset
construction is expensive, you would typically build it only on rank 0 and
broadcast it, but for reproducible in-memory datasets the simpler approach
shown here works correctly.

The `total_loss` accumulation is guarded by `accelerator.sync_gradients`. If
you logged `loss` on every micro-batch step you would count four times as many
steps as actual optimizer updates, making the reported loss artificially low by
a factor of `gradient_accumulation_steps`.

The `scheduler.step()` is placed inside `accumulate()`. Accelerate's wrapped
scheduler is smart enough to step only when `sync_gradients=True`, so even
though the call appears to run every micro-batch, the scheduler only advances
once per actual optimizer step.

---

## 9. DDP Communication Hooks

### 9.1 Why gradient compression exists

In a standard DDP all-reduce, gradients are transmitted at full float32
precision. For a model with 100 million parameters, that is 400 MB of data per
all-reduce round trip. On a machine with NVLink (600 GB/s bidirectional
bandwidth) this takes under 1 ms. On a multi-node cluster connected by 100 Gbps
Ethernet (12.5 GB/s effective), the same all-reduce takes 32 ms — roughly as
long as the backward pass itself.

Communication hooks intercept the gradient tensor just before it enters the
all-reduce and allow you to compress, quantize, or transform it first. The
decompression happens on the receiving end. Compression trades a small amount
of numerical precision for a potentially large reduction in communication time.

### 9.2 The `DDPCommunicationHookType` enum

Defined in `src/accelerate/utils/dataclasses.py`:

```python
class DDPCommunicationHookType(BaseEnum):
    NO             = "no"             # default: no hook
    FP16           = "fp16"           # cast gradients to float16
    BF16           = "bf16"           # cast gradients to bfloat16
    POWER_SGD      = "power_sgd"      # low-rank gradient approximation
    BATCHED_POWER_SGD = "batched_power_sgd"  # batched variant of PowerSGD
```

`FP16` and `BF16` are simple compression hooks. Each gradient bucket is cast to
half-precision, all-reduced, and then cast back to float32. Communication
volume is halved. The trade-off is minor numerical noise in the gradient
average; in practice this rarely affects convergence.

`POWER_SGD` is a low-rank approximation algorithm. It represents each gradient
matrix as the product of two smaller matrices, communicates those smaller
matrices, and reconstructs an approximation of the full gradient on each
process. For tall, thin gradient matrices (common in transformer attention
layers) this can reduce communication volume by 10–100x at the cost of some
additional GPU computation.

### 9.3 Using `DistributedDataParallelKwargs`

```python
from accelerate import Accelerator, DDPCommunicationHookType
from accelerate.utils import DistributedDataParallelKwargs

# Simple FP16 gradient compression
ddp_kwargs = DistributedDataParallelKwargs(
    comm_hook=DDPCommunicationHookType.FP16
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
```

For PowerSGD with a custom approximation rank:

```python
ddp_kwargs = DistributedDataParallelKwargs(
    comm_hook=DDPCommunicationHookType.POWER_SGD,
    comm_state_option={"matrix_approximation_rank": 2}
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
```

You can also combine a primary hook with a compression wrapper. For example,
run PowerSGD with FP16 compression of the low-rank matrices:

```python
ddp_kwargs = DistributedDataParallelKwargs(
    comm_hook=DDPCommunicationHookType.POWER_SGD,
    comm_wrapper=DDPCommunicationHookType.FP16,
    comm_state_option={"matrix_approximation_rank": 1}
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
```

### 9.4 What Accelerate does internally

When `prepare()` wraps the model in DDP, it checks whether any comm hook was
configured via `DistributedDataParallelKwargs`. If so, it calls
`model.register_comm_hook(state, hook_fn)` on the resulting DDP model, where
`hook_fn` is one of the implementations in
`torch.distributed.algorithms.ddp_comm_hooks`. The mapping from
`DDPCommunicationHookType` enum values to PyTorch hook functions lives in the
`prepare_model()` logic inside `accelerator.py`.

The `comm_state_option` dict is passed to the hook's state constructor. For
PowerSGD this populates a `PowerSGDState` object that holds the compression
rank, the error feedback buffers, and the warm-up iteration count.

### 9.5 When communication hooks actually help

Communication hooks help when your training is **communication-bound**: when
the time spent in all-reduce is a significant fraction of total step time. This
is most likely in these scenarios:

- Multi-node training across machines with 10–100 Gbps Ethernet (not NVLink)
- Very large models where gradient tensors are large
- Scenarios where the forward/backward pass is fast relative to model size
  (small inputs, large parameters)

On a single machine with NVLink-connected GPUs, communication hooks typically
provide no measurable benefit because the all-reduce is already faster than the
forward pass. Profile before adding hooks; the additional computation (casting,
matrix decomposition) can sometimes be slower than the communication it
replaces.

---

## 10. `accelerator.wait_for_everyone()`

One utility you will need once you start writing code that runs differently on
different processes is `wait_for_everyone()`, which wraps
`torch.distributed.barrier()`. Every process blocks at this call until all
processes have reached it.

```python
# Save a checkpoint only on rank 0
if accelerator.is_main_process:
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), "checkpoint.pt")

# Wait for rank 0 to finish writing before any process reads the file
accelerator.wait_for_everyone()

# Now safe for all processes to read the checkpoint if needed
```

A common mistake is to write the file and immediately try to read it from
another process without the barrier. The barrier ensures the filesystem write
is complete before any process proceeds.

---

## 11. Exercises

These exercises build progressively on the complete script from Section 8.
Each one deepens a specific aspect of the chapter. Resist the urge to look up
answers first; the goal is to develop the intuition for tracing through what
Accelerate is doing.

**Exercise 1: Observe process isolation**

Add the following line immediately after `accelerator.prepare(...)` and run
with `--num_processes=2`:

```python
print(f"[rank {accelerator.process_index}] I am alive")
```

You should see this printed twice, once from each process. Then replace `print`
with `accelerator.print` and observe that it prints only once. Explain why.

**Exercise 2: Trace the DataLoader transformation**

Add these diagnostic prints before and after `prepare()`:

```python
accelerator.print(f"Before prepare: type={type(train_loader)}")
accelerator.print(f"Before prepare: batch_sampler={type(train_loader.batch_sampler)}")

model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(...)

accelerator.print(f"After prepare: type={type(train_loader)}")
# Note: may need to inspect __class__.__name__ on the wrapped loader
```

Compare the type before and after. Look at the source of `DataLoaderShard`
in `data_loader.py` and identify the `__iter__` method. Find the RNG
synchronization call and the device-placement call.

**Exercise 3: Gradient accumulation correctness**

Modify the training loop to log the step count at which `optimizer.step()` is
actually called (use `accelerator.sync_gradients` as the guard). For a dataset
of 2048 samples with batch size 32 and `gradient_accumulation_steps=4`, how
many actual optimizer updates do you expect per epoch? Verify your count.

**Exercise 4: Evaluation with gather_for_metrics vs gather**

Change `N_EVAL` to a value that is not divisible by 2 (for example, 513).
Compute the evaluation accuracy using both `gather()` and
`gather_for_metrics()`. Compare the results. Explain why they differ and which
is correct.

**Exercise 5: Add FP16 gradient compression**

Add `DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16)`
to the `Accelerator` constructor. Add timing around the training loop
using `time.perf_counter()`. Compare total training time with and without
the hook on a two-GPU machine. On what kind of interconnect would you expect
to see a benefit? What happens if you run this with `--num_processes=1`?

**Exercise 6: Manual vs automatic gradient accumulation**

Rewrite the training loop without using `accumulate()`. Implement the
`no_sync()` guard and the loss division manually. Verify that the final
trained weights are identical (use `torch.allclose` on `state_dict` tensors)
to the version using `accumulate()`. This exercise makes the abstraction
transparent.

---

## 12. What You Should Understand After This Level

You have now worked through every major mechanism that makes multi-GPU DDP
training work in Accelerate. To check your understanding, you should be able
to answer the following questions without looking anything up:

**On process spawning.** `accelerate launch --num_processes=2 train.py` results
in two independent Python processes. What are the names of the four environment
variables each process reads to determine its identity, and where in the
Accelerate source does that reading happen?

**On data distribution.** When `prepare()` rebuilds a map-style DataLoader, it
replaces the `BatchSampler` with a `BatchSamplerShard`. What does
`BatchSamplerShard` do differently from a standard sampler, and what class is
the rebuilt loader an instance of?

**On gradient synchronization.** DDP installs hooks on the model's backward
pass that trigger all-reduce. What does `no_sync()` do, and why is it safe
to call `optimizer.step()` after accumulating gradients through several
`no_sync()` passes? What ensures the gradients are numerically equivalent to
those you would get from training on the combined large batch?

**On the `accumulate()` internals.** Walk through the body of `accumulate()`
step by step. What field of `GradientState` does it consult? What does
`_do_sync()` do? Why is `AcceleratedOptimizer.step()` a no-op on non-sync
steps?

**On gathering.** You have 2 GPUs. Your evaluation dataset has 100 samples.
With `batch_size=32`, each DataLoader iteration covers 32 samples. After two
batches each process has seen 64 local samples. After `gather()`, how many
tensors are in the concatenated result, and how many of those are real data
points versus padding? What does `gather_for_metrics()` do with the extras?

**On communication hooks.** You configure
`comm_hook=DDPCommunicationHookType.FP16`. Describe what happens to the
gradient tensor between the time DDP finishes computing it and the time the
parameter's `.grad` is updated. On what kind of hardware does this help, and
why?

---

## Appendix A: Annotated Source-Reading Tour

The following file locations and approximate line ranges (as of early 2025;
always verify against `main`) are the primary sources for this chapter's
material:

| File | What to read |
|---|---|
| `src/accelerate/commands/launch.py` | `multi_gpu_launcher()`, arg parsing for `--num_processes`, the delegation to `torch.distributed.run` |
| `src/accelerate/state.py` | `PartialState.__init__()` — env var reading; `GradientState` — fields and `_do_sync()` |
| `src/accelerate/data_loader.py` | `prepare_data_loader()` — the dispatch logic; `DataLoaderShard.__iter__()` — RNG sync; `DataLoaderDispatcher.__iter__()` — scatter logic |
| `src/accelerate/accelerator.py` | `prepare_model()` — DDP wrapping; `accumulate()` — no_sync dispatch; `backward()` — loss scaling; `gather()` / `gather_for_metrics()` |
| `src/accelerate/utils/dataclasses.py` | `DDPCommunicationHookType` enum; `DistributedDataParallelKwargs` dataclass |
| `examples/by_feature/gradient_accumulation.py` | Reference implementation |
| `examples/by_feature/ddp_comm_hook.py` | Reference implementation |

When reading `prepare_data_loader()`, pay close attention to the conditional:

```python
if dispatch_batches is None:
    dispatch_batches = isinstance(dataloader.dataset, IterableDataset)

if dispatch_batches:
    return DataLoaderDispatcher(...)
else:
    # Replace batch_sampler with BatchSamplerShard
    new_batch_sampler = BatchSamplerShard(...)
    return DataLoaderShard(
        dataloader.dataset,
        batch_sampler=new_batch_sampler,
        ...
    )
```

This is the exact branching point between dispatch mode and sampler mode.

When reading `accumulate()`, look for how `ExitStack` is used to conditionally
enter `no_sync()`. Notice that `ExitStack` is necessary because the decision
to enter `no_sync()` is made at runtime (depending on the current step), not
at import time.

When reading `GradientState._do_sync()`, trace how it determines whether the
current batch is the last in an accumulation window. The key expression is
approximately:

```python
self.sync_gradients = (
    self.end_of_dataloader                          # last batch of epoch
    or (self.step + 1) % self.num_steps == 0        # end of window
)
```

After Level 2 you have the conceptual and mechanical tools needed to read
almost any standard DDP training script in the wild and understand exactly
which Accelerate abstractions are at work and what they expand into at the
PyTorch level.

---

*End of Level 2.*