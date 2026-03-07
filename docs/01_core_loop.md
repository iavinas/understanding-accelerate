# Accelerate Internals: Level 1 — The Core Training Loop

> **Series:** Contributing to HuggingFace Accelerate — A Developer's Field Guide  
> **Chapter:** 1 of N — Core Training Loop  
> **Source references:** `src/accelerate/accelerator.py`, `src/accelerate/state.py`,  
> `examples/cv_example.py`, `examples/nlp_example.py`

---

## Table of Contents

1. [Why Accelerate Exists](#why-accelerate-exists)
2. [The 5-Line Integration](#the-5-line-integration)
3. [Before and After: The Full Contrast](#before-and-after-the-full-contrast)
4. [What `Accelerator()` Actually Does at Construction Time](#what-accelerator-actually-does-at-construction-time)
5. [The Singleton State System](#the-singleton-state-system)
   - [SharedDict and the Borg Pattern](#shareddict-and-the-borg-pattern)
   - [PartialState](#partialstate)
   - [AcceleratorState](#acceleratorstate)
   - [GradientState](#gradientstate)
   - [DistributedType](#distributedtype)
6. [What `prepare()` Actually Does](#what-prepare-actually-does)
   - [Model Preparation](#model-preparation)
   - [Optimizer Preparation](#optimizer-preparation)
   - [DataLoader Preparation](#dataloader-preparation)
   - [Scheduler Preparation](#scheduler-preparation)
7. [What `backward()` Actually Does](#what-backward-actually-does)
8. [Process Awareness API](#process-awareness-api)
9. [Experiment Tracking with `log()`](#experiment-tracking-with-log)
10. [Full Exercise: CIFAR-10 Classifier](#full-exercise-cifar-10-classifier)
    - [Step 1: Pure PyTorch Baseline](#step-1-pure-pytorch-baseline)
    - [Step 2: Adding the 5 Accelerate Lines](#step-2-adding-the-5-accelerate-lines)
    - [Step 3: Introspecting the Runtime State](#step-3-introspecting-the-runtime-state)
    - [Step 4: Adding Tracking](#step-4-adding-tracking)
11. [Reading the Source: A Guided Tour](#reading-the-source-a-guided-tour)
12. [What You Should Know After This Chapter](#what-you-should-know-after-this-chapter)
13. [Exercises](#exercises)

---

## Why Accelerate Exists

Writing a standard PyTorch training loop for a single GPU is straightforward. When you want to scale it to multiple GPUs, multiple nodes, or add mixed-precision training, the boilerplate explodes:

```python
# The distributed training ceremony you have to perform manually
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler)

    scaler = torch.cuda.amp.GradScaler()   # for fp16

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)           # must call this for proper shuffling!
        for batch in loader:
            batch = batch.to(rank)         # manual device transfer
            optimizer.zero_grad()
            with torch.autocast("cuda"):   # manual autocast for fp16
                loss = model(batch)
            scaler.scale(loss).backward()  # manual gradient scaling
            scaler.step(optimizer)
            scaler.update()

    cleanup()

# You then need torchrun or mp.spawn to launch this
```

That is a lot of infrastructure for every training script. Accelerate's design goal is to reduce all of that to **five lines**, while keeping the training loop itself 100% vanilla PyTorch. This means you stay in control of the loop; Accelerate is not a trainer framework. It is a thin wrapper.

---

## The 5-Line Integration

Here are the only changes you need to make to any standard PyTorch training script:

```python
from accelerate import Accelerator                          # Line 1: import

accelerator = Accelerator()                                 # Line 2: create

model, optimizer, dataloader = accelerator.prepare(         # Line 3: prepare
    model, optimizer, dataloader
)

# ... inside the training loop:
accelerator.backward(loss)                                  # Line 4: backward
```

And optionally, when you save the model, you unwrap it:

```python
accelerator.save_model(model, save_directory)              # Line 5: save
```

That is the complete API surface you need for the standard case. No `model.to(device)`. No `DistributedDataParallel(model)`. No `DistributedSampler`. No gradient scaler management. All of that is absorbed into those four or five lines.

---

## Before and After: The Full Contrast

The contrast below shows exactly how much code Accelerate replaces. The left column is what you would need for a proper multi-GPU training script with fp16. The right column is the Accelerate version that runs identically on 1 GPU, 8 GPUs, TPUs, and CPU.

```python
# ---- BEFORE (manual distributed + fp16) ----     # ---- AFTER (Accelerate) ----

import os                                            from accelerate import Accelerator
import torch.distributed as dist
from torch.nn.parallel import DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):                         accelerator = Accelerator(
    os.environ["MASTER_ADDR"] = "localhost"              mixed_precision="fp16"
    os.environ["MASTER_PORT"] = "12355"              )
    dist.init_process_group("nccl", ...)

model = MyModel()                                    model = MyModel()
model = model.to(rank)                               # (no .to(device))
model = DDP(model, device_ids=[rank])                # (no DDP wrapping)

sampler = DistributedSampler(dataset,                # (no sampler)
    num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, sampler=sampler)        loader = DataLoader(dataset, shuffle=True)

scaler = torch.cuda.amp.GradScaler()                 # (no scaler)

                                                     model, optimizer, loader = (
                                                         accelerator.prepare(
                                                             model, optimizer, loader
                                                         )
                                                     )

for epoch in range(epochs):
    sampler.set_epoch(epoch)                         # (handled automatically)
    for batch in loader:
        batch = batch.to(rank)                       # (handled automatically)
        with torch.autocast("cuda"):                 # (handled automatically)
            loss = model(batch)
        scaler.scale(loss).backward()                accelerator.backward(loss)
        scaler.step(optimizer)                       optimizer.step()
        scaler.update()                              # (handled automatically)
```

The training loop body -- `optimizer.zero_grad()`, forward pass, `backward`, `optimizer.step()` -- stays exactly the same.

---

## What `Accelerator()` Actually Does at Construction Time

When you call `Accelerator()`, the constructor does a significant amount of work before your training loop ever starts. Understanding this initialization is critical for contributing to the library.

```python
# Simplified view of Accelerator.__init__() — accelerator.py
class Accelerator:
    def __init__(
        self,
        device_placement=True,
        split_batches=False,
        mixed_precision=None,           # "no", "fp16", "bf16", "fp8"
        gradient_accumulation_steps=1,
        dataloader_config=None,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        log_with=None,                  # "tensorboard", "wandb", "comet_ml", etc.
        project_dir=None,
        **kwargs
    ):
        # 1. Read ACCELERATE_MIXED_PRECISION env var if not passed explicitly
        # 2. Validate and store plugin configs (DeepSpeed, FSDP, etc.)
        # 3. Initialize AcceleratorState singleton (which initializes PartialState)
        #    This is where torch.distributed.init_process_group() is called
        # 4. Set up mixed precision: create GradScaler for fp16
        # 5. Set up gradient accumulation state via GradientState singleton
        # 6. Initialize experiment trackers if log_with was specified
        # 7. Store lists to track prepared objects: _models, _optimizers, _dataloaders
```

The most consequential step is step 3: creating `AcceleratorState`. This is where the distributed process group is initialized, the device is determined, and the `DistributedType` is selected. Everything after this is based on what was discovered here.

---

## The Singleton State System

The most architecturally unusual part of Accelerate is its state management. Three classes -- `PartialState`, `AcceleratorState`, and `GradientState` -- are all singletons. This is not typical Python; it uses a pattern borrowed from the "Borg" design (shared state, not shared identity).

### SharedDict and the Borg Pattern

```python
# src/accelerate/state.py (simplified)

class SharedDict:
    """A descriptor that holds a shared mutable dict."""
    def __init__(self):
        self._data = {}

    def __get__(self, obj, objtype=None):
        return self._data

    def __set__(self, obj, value):
        self._data.clear()
        self._data.update(value)


class PartialState:
    _shared_state = SharedDict()    # Class-level descriptor

    def __init__(self):
        # This is the key line: every instance shares the SAME __dict__
        self.__dict__ = self._shared_state
        # So all attribute writes go into the shared dict
```

The critical line is `self.__dict__ = self._shared_state`. In Python, `__dict__` is the namespace for instance attributes. By replacing it with a shared object, every instance of `PartialState` reads from and writes to the exact same dictionary. This achieves singleton-like behavior without overriding `__new__`.

```python
# Proof that this is a true singleton
s1 = PartialState()
s2 = PartialState()

s1.custom_value = 42
print(s2.custom_value)   # prints 42
print(s1 is s2)          # prints False  -- different objects...
print(s1.__dict__ is s2.__dict__)  # prints True -- same namespace!
```

This design enables a pattern that is important for contributors to understand: you can call `PartialState()` anywhere in the codebase (in a custom model, a utility function, a sampler) and always get the current distributed configuration without threading it through as an argument.

### `_reset_state()`

```python
# Used exclusively in tests — src/accelerate/state.py
@classmethod
def _reset_state(cls, reset_partial_state=False):
    """Clears the singleton so it can be re-initialized."""
    cls._shared_state._data.clear()
```

If you look at any test in `tests/`, you will see `PartialState._reset_state()` called in `setUp()` or `tearDown()`. Without it, the first test that initializes a process group would poison all subsequent tests. When writing tests for Accelerate, you must reset state between test cases.

---

### PartialState

`PartialState` is the lightweight singleton that tracks distributed environment basics. It can be used without `Accelerator` at all -- useful in inference scripts, or utilities that only need to know "what process am I?".

```python
from accelerate import PartialState

state = PartialState()

print(state.process_index)        # global rank: 0, 1, 2, ... world_size-1
print(state.local_process_index)  # rank within this node: 0, 1, ... local_world_size-1
print(state.num_processes)        # total number of processes (world_size)
print(state.device)               # torch.device: "cuda:0", "cuda:1", "cpu", etc.
print(state.distributed_type)     # DistributedType enum value
print(state.backend)              # "nccl", "gloo", "xla", etc.
print(state.initialized)          # True if __init__ has run at least once
```

The key attributes `PartialState` initializes are determined by reading environment variables set by the process launcher:

```
LOCAL_RANK    -- which GPU on this node
RANK          -- global rank across all nodes
WORLD_SIZE    -- total number of processes
MASTER_ADDR   -- address of the rank-0 process
MASTER_PORT   -- port for rendezvous
```

When you run `accelerate launch --num_processes 4 script.py`, the `accelerate launch` command sets these environment variables and spawns 4 Python processes. Each process starts, creates `PartialState()`, reads its `LOCAL_RANK`, and from that moment knows exactly who it is in the distributed topology.

```python
# src/accelerate/state.py -- PartialState.__init__ (simplified)
def __init__(self, ...):
    if self.initialized:
        return   # singleton: do nothing if already set up

    # Read launch environment variables
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    env_rank = int(os.environ.get("RANK", 0))
    env_world_size = int(os.environ.get("WORLD_SIZE", 1))

    if env_world_size > 1:
        # Multiple processes: initialize the process group
        torch.distributed.init_process_group(backend=backend)
        self.local_process_index = env_local_rank
        self.num_processes = env_world_size
        self.process_index = env_rank
        self.device = torch.device(f"cuda:{env_local_rank}")
    else:
        # Single process: no distributed setup needed
        self.local_process_index = 0
        self.num_processes = 1
        self.process_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

`PartialState` also provides process-aware utilities:

```python
state = PartialState()

# Context manager: only the main process runs first, then others
with state.main_process_first():
    dataset = load_dataset(...)    # rank 0 downloads; others wait

# Context manager: split a list of items across processes
with state.split_between_processes(my_list) as slice:
    results = [process(item) for item in slice]

# Decorator: run a function only on rank 0
@state.on_main_process
def log_metrics(metrics):
    wandb.log(metrics)
```

---

### AcceleratorState

`AcceleratorState` inherits from `PartialState` and adds training-specific configuration. It is always initialized through `Accelerator.__init__()`, not directly by users.

```python
# src/accelerate/state.py -- AcceleratorState adds:
class AcceleratorState(PartialState):
    def __init__(
        self,
        mixed_precision="no",
        deepspeed_plugin=None,
        fsdp_plugin=None,
        ...
        _from_accelerator=False,   # Safety flag: must be True to init
    ):
        if not _from_accelerator:
            raise ValueError(
                "AcceleratorState should only be created by Accelerator."
            )
        super().__init__(...)

        # These are the additions on top of PartialState:
        self.mixed_precision = mixed_precision
        self.deepspeed_plugin = deepspeed_plugin
        self.fsdp_plugin = fsdp_plugin

        # Refine distributed_type based on plugins
        if deepspeed_plugin is not None:
            self.distributed_type = DistributedType.DEEPSPEED
        elif fsdp_plugin is not None:
            self.distributed_type = DistributedType.FSDP

        # Set up GradScaler for fp16
        if mixed_precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
```

The important point is that `AcceleratorState` **overwrites** `distributed_type` if a plugin is passed. A 4-GPU run that started as `DistributedType.MULTI_GPU` becomes `DistributedType.DEEPSPEED` as soon as you pass a `DeepSpeedPlugin`. This is the central dispatch table that `prepare()` reads.

---

### GradientState

`GradientState` is a third singleton, separate from the state hierarchy, focused entirely on gradient accumulation.

```python
from accelerate.state import GradientState

gs = GradientState()
print(gs.sync_gradients)      # True when gradients should be synced this step
print(gs.end_of_dataloader)   # True on the final batch of the epoch
print(gs._num_steps)          # gradient_accumulation_steps value
```

The `sync_gradients` property controls whether DDP actually performs the all-reduce communication. When you use `accelerator.accumulate(model)` and you are in the middle of accumulation steps, `sync_gradients` is `False`, and DDP is told to skip the expensive all-reduce. On the final accumulation step, it flips to `True` and the all-reduce fires. This is the mechanism behind gradient accumulation in distributed training.

---

### DistributedType

The `DistributedType` enum is the central dispatch key. Every `prepare_*()` method switches on it.

```python
# src/accelerate/utils/dataclasses.py
class DistributedType(str, enum.Enum):
    NO          = "no"           # Single process, no distributed training
    MULTI_GPU   = "multi_gpu"    # Standard DDP across GPUs
    MULTI_CPU   = "multi_cpu"    # DDP across CPU nodes (gloo/mpi)
    DEEPSPEED   = "deepspeed"    # DeepSpeed ZeRO optimization
    FSDP        = "fsdp"         # PyTorch Fully Sharded Data Parallel
    XLA         = "xla"          # TPU via torch_xla
    MEGATRON_LM = "megatron_lm"  # Tensor/pipeline parallelism
    # ... other hardware types: MULTI_NPU, MULTI_XPU, etc.
```

You can inspect this at runtime:

```python
accelerator = Accelerator()
print(accelerator.state.distributed_type)
# On a single GPU:       DistributedType.NO
# On 4 GPUs with DDP:    DistributedType.MULTI_GPU
# With DeepSpeed:        DistributedType.DEEPSPEED
```

---

## What `prepare()` Actually Does

`prepare()` is the core of Accelerate. It accepts any combination of PyTorch objects and dispatches each to a type-specific preparation function.

```python
# src/accelerate/accelerator.py -- prepare() (simplified pseudocode)
def prepare(self, *args, device_placement=None):
    result = []
    for obj in args:
        if isinstance(obj, torch.nn.Module):
            result.append(self._prepare_model(obj))
        elif isinstance(obj, torch.optim.Optimizer):
            result.append(self._prepare_optimizer(obj))
        elif isinstance(obj, DataLoader):
            result.append(self._prepare_data_loader(obj))
        elif isinstance(obj, LRScheduler):
            result.append(self._prepare_scheduler(obj))
        else:
            result.append(obj)  # pass through anything else unchanged
    return tuple(result)
```

The order of arguments does not matter to `prepare()`. You can pass them in any sequence and unpack them in any sequence -- as long as the types match up. You just need to unpack in the same order you passed them in:

```python
# These are equivalent
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
loader, model, optimizer = accelerator.prepare(loader, model, optimizer)
```

### Model Preparation

```python
# src/accelerate/accelerator.py -- prepare_model() (simplified)
def prepare_model(self, model, device_placement=True):
    distributed_type = self.state.distributed_type

    # 1. Move to device (unless device_placement=False)
    if device_placement:
        model = model.to(self.device)

    # 2. Wrap based on distributed type
    if distributed_type == DistributedType.NO:
        pass  # single process: do nothing, just device placement

    elif distributed_type == DistributedType.MULTI_GPU:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(
            model,
            device_ids=[self.local_process_index],
            output_device=self.local_process_index,
        )

    elif distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel
        # Uses fsdp_plugin configuration for sharding strategy,
        # auto_wrap_policy, cpu_offload, etc.
        model = FullyShardedDataParallel(model, ...)

    elif distributed_type == DistributedType.DEEPSPEED:
        # DeepSpeed initialization happens later, when the optimizer
        # is also available: model + optimizer are initialized together
        pass

    # Track the model for save/load operations
    self._models.append(model)
    return model
```

The DDP wrapping is what makes gradients synchronize automatically. When `loss.backward()` fires on a DDP-wrapped model, PyTorch registers gradient hooks that trigger an `all-reduce` operation: gradients are summed (then divided by `world_size`) across all processes. By the time `optimizer.step()` runs, every process has identical averaged gradients and produces identical weight updates. This is data parallelism.

**Important: unwrapping for save.** After `prepare()`, your model variable holds a `DistributedDataParallel` wrapper. To save the actual model weights, you need to unwrap it:

```python
# Wrong: saves DDP wrapper state dict with "module." prefixes
torch.save(model.state_dict(), "model.pt")

# Correct: unwraps DDP/FSDP/DeepSpeed wrapper first
accelerator.save_model(model, "save_dir/")
# or: unwrapped = accelerator.unwrap_model(model)
```

### Optimizer Preparation

For standard DDP and single-process training, the optimizer is wrapped in `AcceleratedOptimizer`, a thin proxy that handles:

1. **Gradient scaling for fp16**: Before `optimizer.step()`, it calls `scaler.unscale_(optimizer)` to convert scaled gradients back to true gradients.
2. **Gradient synchronization gating**: Checks `GradientState.sync_gradients` before stepping. During gradient accumulation, optimizer steps are skipped on intermediate sub-steps.
3. **Device placement of optimizer state**: Ensures optimizer state (Adam's first/second moments, etc.) is placed on the correct device.

```python
# src/accelerate/optimizer.py -- AcceleratedOptimizer.step() (simplified)
def step(self, closure=None):
    if not self.gradient_state.sync_gradients:
        # We are in the middle of gradient accumulation; do not step yet
        return

    if self.scaler is not None:
        # fp16: unscale first, then step through scaler
        self.scaler.step(self.optimizer, closure)
        self.scaler.update()
    else:
        self.optimizer.step(closure)
```

For **DeepSpeed**, the optimizer is replaced entirely. `deepspeed.initialize()` is called with both the model and optimizer, and returns a DeepSpeed engine. In this case, the optimizer you passed to `prepare()` is no longer used; the engine manages it internally.

### DataLoader Preparation

DataLoader preparation is arguably the most complex part of `prepare()`. When you call `prepare()` on a DataLoader, Accelerate replaces it with a `DataLoaderShard` (the default) or a `DataLoaderDispatcher`.

**DataLoaderShard (default -- "sharding" strategy):**

Each process gets a shard of the dataset. Accelerate does not simply inject a `DistributedSampler`; it does something more sophisticated. It wraps the existing sampler (whether `RandomSampler`, `SequentialSampler`, or a custom sampler) with a `SeedableRandomSampler` that adds:

- Epoch-aware seeding so shuffling is consistent and non-duplicating across processes.
- No requirement to manually call `sampler.set_epoch(epoch)` -- the prepared DataLoader handles this automatically.

```python
# src/accelerate/data_loader.py -- prepare_data_loader() (simplified)
def prepare_data_loader(dataloader, device, process_index, num_processes, ...):
    if num_processes == 1:
        # Single process: just move batches to device, no sharding needed
        return DataLoaderShard(dataloader, device=device)

    # Multiple processes: shard the dataset
    # Replace the sampler with a sharded variant
    new_batch_sampler = BatchSamplerShard(
        dataloader.batch_sampler,
        num_processes=num_processes,
        process_index=process_index,
        split_batches=split_batches,
    )

    # Create a new DataLoader with the sharded sampler
    new_dataloader = DataLoader(
        dataloader.dataset,
        batch_sampler=new_batch_sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
    )

    # Wrap in DataLoaderShard to auto-move batches to device
    return DataLoaderShard(new_dataloader, device=device, ...)
```

The `DataLoaderShard` auto-moves every batch to `accelerator.device` as batches are yielded. This is why you do not need `.to(device)` inside the training loop.

**DataLoaderDispatcher (dispatch strategy):**

This is an alternative mode enabled by `DataLoaderConfiguration(dispatch_batches=True)`. In dispatch mode, only the main process (rank 0) actually iterates the DataLoader. It then broadcasts each batch to all other processes. This is useful when:

- Your DataLoader has stateful side effects that should only happen once.
- You cannot shard the dataset (e.g., it is a streaming dataset with no `len()`).
- Your batch generation logic is too complex to shard cleanly.

```python
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

accelerator = Accelerator(
    dataloader_config=DataLoaderConfiguration(dispatch_batches=True)
)
```

### Scheduler Preparation

Learning rate schedulers are wrapped in `AcceleratedScheduler`:

```python
# src/accelerate/scheduler.py -- AcceleratedScheduler.step() (simplified)
def step(self):
    if not self.gradient_state.sync_gradients:
        # Skip scheduler step during gradient accumulation sub-steps
        return
    self.scheduler.step()
```

This ensures the scheduler advances exactly once per "real" optimizer step, even when gradient accumulation is in use. Without this, with `gradient_accumulation_steps=4`, the scheduler would step 4 times for every actual optimizer update.

---

## What `backward()` Actually Does

```python
# src/accelerate/accelerator.py -- backward() (simplified)
def backward(self, loss, **kwargs):
    distributed_type = self.state.distributed_type

    if distributed_type == DistributedType.DEEPSPEED:
        # DeepSpeed manages its own backward pass through the engine
        self.deepspeed_engine_wrapped.backward(loss, **kwargs)

    elif self.scaler is not None:
        # fp16 mixed precision: scale the loss before backward
        # to prevent gradient underflow from small fp16 values
        self.scaler.scale(loss).backward(**kwargs)

    else:
        # Standard case: plain backward, nothing special
        loss.backward(**kwargs)
```

The function has three branches, and understanding why each exists is important:

**Branch 1: DeepSpeed.** DeepSpeed replaces the standard backward pass entirely. Its `engine.backward()` handles gradient clipping, gradient accumulation, and ZeRO's parameter sharding all internally. Calling `loss.backward()` directly on a DeepSpeed model would bypass all of that.

**Branch 2: fp16 with GradScaler.** In half-precision training, gradient values can be so small they underflow to zero in fp16's limited range (minimum positive normal value: ~6e-5). The `GradScaler` multiplies the loss by a large scale factor before backward, amplifying the gradients. After the backward pass, `scaler.unscale_()` in `AcceleratedOptimizer.step()` divides them back down before the optimizer update. The scale factor is dynamically adjusted: it grows if no gradient overflow is detected, and shrinks if overflow (inf/nan gradients) is detected.

**Branch 3: Standard.** For `no`, `bf16`, and multi-GPU DDP, `loss.backward()` is all that is needed. DDP has already registered gradient hooks on the model that fire the all-reduce automatically when `backward()` reaches them. BF16 does not need scaling because it has a wider dynamic range (same exponent bits as FP32).

```python
# The gradient flow in fp16 DDP:
#
#  loss (fp32)
#    |
#    v
#  scaler.scale(loss)  -->  loss * scale_factor  (still fp32 in the computation graph)
#    |
#    v
#  .backward()         -->  gradients computed in fp16, scaled up
#    |                     DDP all-reduce fires here (on scaled gradients)
#    v
#  scaler.unscale_()   -->  divide gradients by scale_factor
#    |
#    v
#  optimizer.step()    -->  weight update in fp32 master weights
#    |
#    v
#  scaler.update()     -->  adjust scale factor for next step
```

---

## Process Awareness API

When running distributed training, many operations should only happen on one process. Accelerate provides a clean API for this.

```python
accelerator = Accelerator()

# --- Print only from rank 0 ---
print("This prints from EVERY process -- can cause garbled output")
accelerator.print("This prints ONCE, from rank 0 only")

# --- Booleans ---
accelerator.is_main_process          # True on global rank 0
accelerator.is_local_main_process    # True on local rank 0 (rank 0 per node)
accelerator.num_processes            # total number of processes
accelerator.process_index            # current global rank
accelerator.local_process_index      # current local rank

# --- Barriers ---
accelerator.wait_for_everyone()      # blocks until all processes reach this line

# --- Context manager: rank 0 goes first, then others ---
with accelerator.main_process_first():
    # Useful for dataset downloads, tokenization, file writes
    # Rank 0 runs this block; all other ranks wait
    # Then all ranks run the block (rank 0's cached result is available)
    tokenized = dataset.map(tokenize_fn)

# --- Gather across processes (for metrics) ---
all_predictions = accelerator.gather(my_predictions)  # concat from all processes
all_predictions = accelerator.gather_for_metrics(my_predictions)  # also handles padding
```

The difference between `is_main_process` and `is_local_main_process` matters in multi-node setups:

```
Node 0:  rank 0 (local rank 0)  <- is_main_process AND is_local_main_process
         rank 1 (local rank 1)
         rank 2 (local rank 2)
         rank 3 (local rank 3)
Node 1:  rank 4 (local rank 0)  <- is_local_main_process (but NOT is_main_process)
         rank 5 (local rank 1)
         rank 6 (local rank 2)
         rank 7 (local rank 3)
```

Use `is_local_main_process` when you want one process per node to do something (e.g., write to local disk, log to a local file). Use `is_main_process` when only one process in the entire job should do something (e.g., save the final model, log to a remote tracker).

---

## Experiment Tracking with `log()`

Accelerate provides a unified logging API that routes to any supported backend: TensorBoard, Weights & Biases, Comet ML, MLflow, ClearML, and others.

```python
# Setup
accelerator = Accelerator(log_with="tensorboard", project_dir="./logs")
accelerator.init_trackers(
    project_name="cifar10_experiment",
    config={"learning_rate": 1e-3, "batch_size": 64, "epochs": 10},
)

# Inside the training loop
accelerator.log(
    {
        "train/loss": loss.item(),
        "train/accuracy": accuracy,
        "learning_rate": optimizer.param_groups[0]["lr"],
    },
    step=global_step,
)

# At the end of training
accelerator.end_training()
```

`accelerator.log()` only logs from the main process. You do not need to guard it with `if accelerator.is_main_process`. The tracker handles that internally.

For TensorBoard specifically, the logs are written to `{project_dir}/runs/{current_datetime}` and can be viewed with:

```bash
tensorboard --logdir ./logs
```

---

## Full Exercise: CIFAR-10 Classifier

This exercise walks through building a CIFAR-10 classifier in four stages: pure PyTorch, adding Accelerate, introspecting the runtime, and adding experiment tracking.

### Step 1: Pure PyTorch Baseline

```python
# cifar10_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --- Model ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = self.pool(F.relu(self.conv3(x)))  # 8 -> 4
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x


# --- Data ---

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    val_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


# --- Training loop ---

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2,
        steps_per_epoch=len(train_loader), epochs=10
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # manual device

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()                                          # plain backward
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d} | loss={avg_loss:.4f} | acc={train_acc:.3f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"           val_acc={val_acc:.3f}")


if __name__ == "__main__":
    train()
```

Run this with:

```bash
python cifar10_baseline.py
```

It downloads CIFAR-10 and trains for 10 epochs. Verify it reaches roughly 75-80% validation accuracy.

---

### Step 2: Adding the 5 Accelerate Lines

Now we convert the baseline to use Accelerate. The diff is intentionally small:

```python
# cifar10_accelerate.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator                             # LINE 1: import


class SimpleCNN(nn.Module):
    # (identical to above -- no changes needed in the model definition)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)


def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    val_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    # Note: no sampler, no .to(device). shuffle=True is fine.
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def train():
    accelerator = Accelerator()                                # LINE 2: create

    train_loader, val_loader = get_dataloaders()

    model = SimpleCNN()                                        # no .to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2,
        steps_per_epoch=len(train_loader), epochs=10
    )
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, scheduler = accelerator.prepare(  # LINE 3
        model, optimizer, train_loader, scheduler
    )
    # val_loader is not prepared -- it is used only on the main process
    # (or we can prepare it too, which enables distributed eval)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # No .to(device) needed: DataLoaderShard handles it automatically
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)                         # LINE 4: backward
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        accelerator.print(f"Epoch {epoch+1:02d} | loss={avg_loss:.4f} | acc={train_acc:.3f}")

        # Validation (run on all processes, gather results)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(accelerator.device)
                labels = labels.to(accelerator.device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        accelerator.print(f"           val_acc={val_acc:.3f}")

    # Save the final model (unwraps DDP/FSDP before saving)     # LINE 5
    accelerator.save_model(model, "cifar10_model")


if __name__ == "__main__":
    train()
```

Run as a single process (behaves identically to the baseline):

```bash
python cifar10_accelerate.py
```

Run on 4 GPUs (no code change):

```bash
accelerate launch --num_processes 4 cifar10_accelerate.py
```

Run with fp16 mixed precision on 4 GPUs:

```bash
accelerate launch --num_processes 4 --mixed_precision fp16 cifar10_accelerate.py
```

---

### Step 3: Introspecting the Runtime State

Add the following block right after `accelerator = Accelerator()` to see exactly what was auto-detected:

```python
def train():
    accelerator = Accelerator()

    # Print a summary of what Accelerate detected and configured.
    # accelerator.print() ensures this only appears once, from rank 0.
    accelerator.print("=" * 60)
    accelerator.print("Accelerate Runtime Configuration")
    accelerator.print("=" * 60)
    accelerator.print(f"  device:           {accelerator.device}")
    accelerator.print(f"  num_processes:    {accelerator.num_processes}")
    accelerator.print(f"  process_index:    {accelerator.process_index}")
    accelerator.print(f"  local_proc_idx:   {accelerator.local_process_index}")
    accelerator.print(f"  is_main_process:  {accelerator.is_main_process}")
    accelerator.print(f"  distributed_type: {accelerator.state.distributed_type}")
    accelerator.print(f"  mixed_precision:  {accelerator.mixed_precision}")
    accelerator.print(f"  use_distributed:  {accelerator.use_distributed}")
    accelerator.print("=" * 60)

    # ... rest of training
```

Expected output on a single GPU:

```
============================================================
Accelerate Runtime Configuration
============================================================
  device:           cuda:0
  num_processes:    1
  process_index:    0
  local_proc_idx:   0
  is_main_process:  True
  distributed_type: DistributedType.NO
  mixed_precision:  no
  use_distributed:  False
============================================================
```

Expected output on 4 GPUs with `accelerate launch --num_processes 4`:

```
============================================================
Accelerate Runtime Configuration
============================================================
  device:           cuda:0
  num_processes:    4
  process_index:    0
  local_proc_idx:   0
  is_main_process:  True
  distributed_type: DistributedType.MULTI_GPU
  mixed_precision:  no
  use_distributed:  True
============================================================
```

You can also inspect what `prepare()` actually returned:

```python
# After prepare():
print(type(model))
# Single GPU:  <class 'cifar10_accelerate.SimpleCNN'>
# Multi GPU:   <class 'torch.nn.parallel.distributed.DistributedDataParallel'>

print(type(optimizer))
# Always:      <class 'accelerate.optimizer.AcceleratedOptimizer'>

print(type(train_loader))
# Always:      <class 'accelerate.data_loader.DataLoaderShard'>
```

---

### Step 4: Adding Tracking

```python
# cifar10_tracked.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator


class SimpleCNN(nn.Module):
    # (same as before)
    ...


def get_dataloaders(batch_size=128):
    # (same as before)
    ...


def train():
    config = {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 10,
        "max_lr": 1e-2,
        "weight_decay": 1e-4,
        "architecture": "SimpleCNN",
    }

    accelerator = Accelerator(
        log_with="tensorboard",    # or "wandb", "comet_ml", "mlflow"
        project_dir="./tb_logs",
        mixed_precision="bf16",    # try bf16 for modern GPUs
    )

    # Initialize trackers BEFORE training starts.
    # config dict is saved as hyperparameters in TensorBoard HParams.
    accelerator.init_trackers(
        project_name="cifar10_level1",
        config=config,
    )

    train_loader, val_loader = get_dataloaders(batch_size=config["batch_size"])

    model = SimpleCNN()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    num_training_steps = config["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        total_steps=num_training_steps,
    )
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    global_step = 0

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            global_step += 1

            # Log every 50 steps
            if global_step % 50 == 0:
                accelerator.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

        train_acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / len(train_loader)

        # Per-epoch logging
        accelerator.log(
            {
                "train/epoch_loss": avg_loss,
                "train/epoch_accuracy": train_acc,
                "epoch": epoch + 1,
            },
            step=global_step,
        )

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(accelerator.device)
                labels = labels.to(accelerator.device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        accelerator.log({"val/accuracy": val_acc}, step=global_step)
        accelerator.print(
            f"Epoch {epoch+1:02d} | "
            f"loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f}"
        )

    accelerator.end_training()
    accelerator.save_model(model, "cifar10_final")


if __name__ == "__main__":
    train()
```

View the TensorBoard dashboard:

```bash
tensorboard --logdir ./tb_logs
```

---

## Reading the Source: A Guided Tour

Now that you have a mental model of what `prepare()` and `backward()` do, here is a suggested reading path through the source code. Follow this order to build understanding incrementally.

**1. `src/accelerate/state.py` -- Lines 92-340**

Start with `SharedDict`, then read `PartialState.__init__`. Focus on:
- How `self.__dict__ = self._shared_state` achieves the singleton.
- The `if self.initialized: return` early exit.
- The `LOCAL_RANK` / `WORLD_SIZE` environment variable reading.
- The `_prepare_backend()` method and how it selects `DistributedType`.

**2. `src/accelerate/state.py` -- Lines 708-1000**

Read `AcceleratorState.__init__`. Focus on:
- How `_from_accelerator=True` is enforced.
- How `distributed_type` is refined based on plugins.
- How `GradScaler` is set up for fp16.

**3. `src/accelerate/accelerator.py` -- Lines 280-640**

Read `Accelerator.__init__`. Focus on:
- The order of initialization (plugins before state, state before scaler).
- How `log_with` initializes trackers.
- The lists `self._models`, `self._optimizers`, `self._dataloaders` -- these are what `save_state()` and `load_state()` iterate over.

**4. `src/accelerate/accelerator.py` -- `prepare()` and `prepare_model()`**

Search for `def prepare(` and `def prepare_model(`. Read:
- The type dispatch loop in `prepare()`.
- The `distributed_type` switch in `prepare_model()`.
- The DDP wrapping code and where `device_ids` comes from.

**5. `src/accelerate/accelerator.py` -- `backward()`**

Search for `def backward(`. It is short (about 20 lines). Read every branch.

**6. `src/accelerate/optimizer.py` -- `AcceleratedOptimizer.step()`**

Read the full `step()` method. Focus on:
- The `sync_gradients` check.
- The scaler path vs. the plain path.

**7. `src/accelerate/data_loader.py` -- `prepare_data_loader()`**

This is the most complex function in the library. Start with the function signature and read the top-level branching logic (sharding vs. dispatching). You do not need to understand every line on the first pass.

**8. `examples/nlp_example.py`**

Read the full example. Notice:
- `accelerator.prepare()` is called with model, optimizer, train_loader, eval_loader, and scheduler all at once.
- `accelerator.gather_for_metrics()` is used during evaluation to collect predictions from all processes before computing accuracy.
- `accelerator.print()` is used instead of `print()`.

---

## What You Should Know After This Chapter

By the end of this chapter you should be able to answer these questions without looking anything up:

**Q: What does `prepare()` return for a `torch.nn.Module` on 4 GPUs?**
A: A `DistributedDataParallel`-wrapped version of the model, placed on the process's local GPU. The original module is accessible via `accelerator.unwrap_model(model)`.

**Q: Why does `prepare()` accept multiple objects at once?**
A: Because the DeepSpeed initialization requires both the model and optimizer simultaneously (`deepspeed.initialize(model, optimizer, ...)`). Passing them separately would make that impossible. The type dispatch handles each one individually, except for the DeepSpeed joint initialization.

**Q: What does `accelerator.backward(loss)` do differently from `loss.backward()`?**
A: For plain DDP and BF16 it is identical. For FP16 it applies GradScaler: scales the loss before backward to prevent underflow, then the scaler unscales before `optimizer.step()`. For DeepSpeed it delegates to the engine's backward method.

**Q: Why does `PartialState` use `self.__dict__ = self._shared_state`?**
A: To implement the singleton pattern without overriding `__new__`. All instances of `PartialState` share the same attribute dictionary, so reading or writing any attribute on any instance affects all instances. This lets library internals call `PartialState()` anywhere and always get the current distributed configuration.

**Q: What is the difference between `accelerator.print()` and `print()`?**
A: `accelerator.print()` only executes on `is_main_process` (rank 0). In a 4-process run, plain `print()` would output the same line 4 times (once per process), often interleaved and garbled. `accelerator.print()` prints once.

**Q: What happens to the DataLoader's sampler after `prepare()`?**
A: The original sampler is replaced with a `BatchSamplerShard` that divides the dataset across processes. Each process sees `1/num_processes` of the dataset per epoch, non-overlapping. The shard also handles epoch-based seeding automatically, so you do not need to call `sampler.set_epoch(epoch)` yourself.

**Q: When would you use `gather_for_metrics()` instead of `gather()`?**
A: When the dataset size is not evenly divisible by `num_processes`. `prepare_data_loader()` by default pads the last batch so all processes receive equal-sized batches. `gather_for_metrics()` is aware of this padding and removes the extra samples before you compute your metrics. `gather()` just concatenates everything, including the padding.

---

## Exercises

These exercises are designed to build muscle memory for reading the source and for writing new contributors' tests.

**Exercise 1: Confirm the singleton.**

Write a script that creates two `Accelerator()` instances and verifies they share state:

```python
from accelerate import Accelerator
from accelerate.state import PartialState

acc1 = Accelerator()
acc2 = Accelerator()

# Both should point to the same state object
assert acc1.state is acc2.state, "State objects should be identical"
print("OK: acc1.state is acc2.state")

# PartialState is also the same
ps1 = PartialState()
ps2 = PartialState()
assert ps1.__dict__ is ps2.__dict__, "Should share __dict__"
print("OK: PartialState instances share __dict__")
```

**Exercise 2: Trace through prepare() manually.**

Without running code, trace what `accelerator.prepare(model, optimizer, loader)` calls on a single GPU. Write the call chain as pseudocode comments:

```
prepare(model, optimizer, loader)
  -> isinstance(model, nn.Module): True
     -> prepare_model(model)
        -> distributed_type == NO: device_placement only
        -> model.to(self.device)
        -> self._models.append(model)
        -> return model
  -> isinstance(optimizer, Optimizer): True
     -> prepare_optimizer(optimizer)
        -> wrap in AcceleratedOptimizer(optimizer, scaler=None)
        -> return AcceleratedOptimizer
  -> isinstance(loader, DataLoader): True
     -> prepare_data_loader(loader)
        -> num_processes == 1: DataLoaderShard (no sharding needed)
        -> return DataLoaderShard(loader, device=self.device)
```

**Exercise 3: Add gradient accumulation.**

Modify `cifar10_accelerate.py` to use 4 gradient accumulation steps:

```python
accelerator = Accelerator(gradient_accumulation_steps=4)

for images, labels in train_loader:
    with accelerator.accumulate(model):    # manages sync_gradients automatically
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
```

Observe that `optimizer.step()` is called every iteration but `AcceleratedOptimizer` internally skips the actual step when `sync_gradients` is False. Add a `print(accelerator.sync_gradients)` to see it toggle.

**Exercise 4: Read a test.**

Open `tests/test_accelerator.py`. Find a test that calls `PartialState._reset_state()`. Read what state it sets up, what it tests, and why `_reset_state()` is necessary before and after. Write a comment explaining why removing `_reset_state()` would cause the test to fail or, worse, to pass incorrectly.

**Exercise 5: Implement gather_and_reduce.**

Using only `accelerator.gather()` and standard PyTorch operations, implement a function that computes global accuracy across all processes without using `gather_for_metrics()`:

```python
def compute_global_accuracy(accelerator, local_correct, local_total):
    """
    Gather correct counts and total counts from all processes,
    then compute global accuracy on the main process.
    """
    # Hint: wrap scalars in tensors first
    local_correct_t = torch.tensor(local_correct, device=accelerator.device)
    local_total_t   = torch.tensor(local_total,   device=accelerator.device)

    all_correct = accelerator.gather(local_correct_t)   # shape: (num_processes,)
    all_total   = accelerator.gather(local_total_t)     # shape: (num_processes,)

    if accelerator.is_main_process:
        global_acc = all_correct.sum().item() / all_total.sum().item()
        return global_acc
    return None
```

Then compare the result to what `gather_for_metrics()` + standard accuracy computation gives you. They should agree when the dataset is evenly divisible by `num_processes`.

---

*End of Chapter 1. In Chapter 2 we will go deeper into mixed precision training: how GradScaler chooses its scale factor, what happens when it detects inf/nan gradients, and how BF16 achieves stability without scaling.*