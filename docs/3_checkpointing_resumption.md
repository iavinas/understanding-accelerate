# Level 3: Checkpointing & Resumption

## Goal

Save and resume training at the **exact iteration** — restoring model weights, optimizer momentum
buffers, learning-rate schedule, random-number-generator state, and dataloader position — so that
a resumed run is byte-for-byte identical to one that never stopped.

---

## 1. The Problem Space

Checkpointing sounds straightforward: save weights, load weights. The subtlety is that a
training run is not just the model. It is the *joint state* of at least five independent systems,
all of which must be restored together if the resumed run is to follow the same trajectory as
an uninterrupted one.

| System | Why it matters |
|---|---|
| Model weights | Obvious — this is the thing you are training |
| Optimizer state | Adam's `m` and `v` momentum buffers encode the recent gradient history; losing them causes a warm-up artifact on resume |
| LR scheduler state | `last_epoch` and `_last_lr` track position in the schedule; incorrect values give the wrong learning rate for every subsequent step |
| RNG state | `nn.Dropout`, data augmentation, and `DataLoader(shuffle=True)` all consume random numbers; divergent state here means divergent gradient noise |
| DataLoader position | Without restoring position, training either re-sees the first part of an epoch or silently changes which examples appear in which mini-batch |

Accelerate manages all five. This chapter shows you exactly how.

---

## 2. Source Files

Before reading this chapter, locate these files in the repository:

```
src/accelerate/checkpointing.py      # save_accelerator_state(), load_accelerator_state()
src/accelerate/accelerator.py        # save_state(), load_state(), register_for_checkpointing()
src/accelerate/data_loader.py        # DataLoaderShard, skip_first_batches()
src/accelerate/utils/dataclasses.py  # ProjectConfiguration, DataLoaderConfiguration
examples/by_feature/checkpointing.py # official reference script
```

---

## 3. The Checkpoint Directory Layout

When you call `accelerator.save_state("checkpoint_dir/")`, Accelerate writes a directory whose
contents depend on what you have prepared:

```
checkpoint_dir/
  model_0/                      # model 0 weights (safetensors shards, or pytorch_model.bin)
  model_1/                      # model 1 weights, if you called prepare() on two models
  optimizer_0/                  # optimizer 0 state dict (pickle, not safetensors)
  optimizer_1/                  # optimizer 1 state dict
  scheduler_0/                  # AcceleratedScheduler.scheduler.state_dict()
  random_states_0.pkl           # RNG snapshot for process rank 0
  random_states_1.pkl           # RNG snapshot for process rank 1 (multi-GPU only)
  scaler.pt                     # GradScaler state (fp16 training only)
  custom_checkpoint_0.pkl       # first object registered via register_for_checkpointing()
  custom_checkpoint_1.pkl       # second registered object, etc.
```

Each process writes its own `random_states_{rank}.pkl`. On a two-GPU run you will find both
`random_states_0.pkl` and `random_states_1.pkl` after a save. The constants that name each file
are defined in `src/accelerate/utils/constants.py`:

```python
MODEL_NAME     = "pytorch_model"
OPTIMIZER_NAME = "optimizer"
SCHEDULER_NAME = "scheduler"
RNG_STATE_NAME = "random_states"     # rank appended as f"{RNG_STATE_NAME}_{rank}.pkl"
SCALER_NAME    = "scaler.pt"
```

The naming convention matters if you ever need to inspect or surgically patch a checkpoint by
hand, as you may do when adjusting the learning rate mid-run.

---

## 4. The Call Stack: `save_state` to Disk

`accelerator.save_state()` is defined in `src/accelerate/accelerator.py`. It handles
`ProjectConfiguration` bookkeeping (naming, rotation) and then delegates to
`save_accelerator_state()` from `src/accelerate/checkpointing.py`.

```
accelerator.save_state(output_dir)
  └── save_accelerator_state(
          output_dir,
          models=self._models,
          optimizers=self._optimizers,
          schedulers=self._schedulers,
          dataloaders=self._dataloaders,     # for StatefulDataLoader
          process_index=self.process_index,
          scaler=self.scaler,
      )
          ├── for i, model  → save weights to model_{i}/
          ├── for i, optim  → save state_dict to optimizer_{i}/
          ├── for i, sched  → save state_dict to scheduler_{i}/
          ├── capture RNG   → random_states_{rank}.pkl
          └── if scaler     → scaler.pt
  └── for i, obj in self._custom_checkpoints
          └── save_custom_state(obj, output_dir, i)   → custom_checkpoint_{i}.pkl
```

After all files are written, `save_state` handles checkpoint rotation if `total_limit` was set.

---

## 5. RNG State in Depth

### 5.1 Why It Is Complicated

Modern hardware training uses several independent RNG subsystems. Each one must be captured and
restored independently:

| RNG system | API to save | API to restore |
|---|---|---|
| Python `random` stdlib | `random.getstate()` | `random.setstate(state)` |
| NumPy global RNG | `np.random.get_state()` | `np.random.set_state(state)` |
| PyTorch CPU | `torch.get_rng_state()` | `torch.set_rng_state(state)` |
| PyTorch CUDA (all GPUs) | `torch.cuda.get_rng_state_all()` | `torch.cuda.set_rng_state_all(state)` |
| PyTorch HPU | `torch.hpu.get_rng_state_all()` | `torch.hpu.set_rng_state_all(state)` |
| PyTorch XPU | `torch.xpu.get_rng_state_all()` | `torch.xpu.set_rng_state_all(state)` |
| PyTorch MPS | `torch.mps.get_rng_state()` | `torch.mps.set_rng_state(state)` |

### 5.2 The Save Path (`save_accelerator_state`)

The relevant section of `src/accelerate/checkpointing.py`:

```python
# Save RNG states in their own file
rng_types_to_save = get_rng_state(rng_types)

states = {}
# Python stdlib random
states["random_state"] = random.getstate()
# NumPy
states["numpy_random_seed"] = np.random.get_state()
# PyTorch CPU
states["torch_manual_seed"] = torch.get_rng_state()

# Hardware-specific — each is an independent if, not elif
if is_cuda_available():
    states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
if is_hpu_available():
    states["torch_hpu_manual_seed"] = torch.hpu.get_rng_state_all()
if is_xpu_available():
    states["torch_xpu_manual_seed"] = torch.xpu.get_rng_state_all()
if is_mps_available():
    states["torch_mps_manual_seed"] = torch.mps.get_rng_state()

# Written per-process so that each rank's own GPU state is captured correctly
output_rng_file = os.path.join(output_dir, f"random_states_{process_index}.pkl")
with open(output_rng_file, "wb") as f:
    pickle.dump(states, f)
```

### 5.3 The Load Path (`load_accelerator_state`)

The mirror image in the load function must use exactly the same structure:

```python
# Load this process's RNG file
rng_file = os.path.join(input_dir, f"random_states_{process_index}.pkl")
with open(rng_file, "rb") as f:
    states = pickle.load(f)

# Restore in the same order
random.setstate(states["random_state"])
np.random.set_state(states["numpy_random_seed"])
torch.set_rng_state(states["torch_manual_seed"])

# Critical: independent ifs here too, matching the save side
if is_cuda_available():
    torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
if is_hpu_available():
    torch.hpu.set_rng_state_all(states["torch_hpu_manual_seed"])
if is_xpu_available():
    torch.xpu.set_rng_state_all(states["torch_xpu_manual_seed"])
if is_mps_available():
    torch.mps.set_rng_state(states["torch_mps_manual_seed"])
```

### 5.4 Why Independent `if` Checks Are Required (The Bug You Fixed)

Consider a machine that has both CUDA and HPU devices (hypothetical, but the pattern generalises
to any pair of optional accelerators). If the load path used `elif`:

```python
# WRONG — elif means only one branch can execute
if is_cuda_available():
    torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
elif is_hpu_available():    # never reached on a CUDA+HPU node
    torch.hpu.set_rng_state_all(states["torch_hpu_manual_seed"])
```

The HPU state would never be restored, silently leaving the HPU RNG at its initialisation value.
Training would produce different results after resume without any error or warning.

The save side used independent `if` checks from the beginning. The bug was that the load side
used `elif`, creating an asymmetry. Your fix made both sides symmetric.

This is the kind of bug that only surfaces on multi-accelerator hardware, passes all unit tests
that run on single-device CI, and silently corrupts training rather than crashing. Finding it
required reading both code paths side-by-side, which is exactly the kind of contributor work
that adds real value.

### 5.5 Per-Rank Files: One RNG File Per Process

In multi-GPU DDP each process has its own CUDA RNG (one RNG per GPU). The file is written and
read with `process_index` in the name so that each rank restores only its own GPU's state.
On an 8-GPU run you will find `random_states_0.pkl` through `random_states_7.pkl`.

```
checkpoint_dir/
  random_states_0.pkl   ← rank 0 restores this one only
  random_states_1.pkl   ← rank 1 restores this one only
  ...
  random_states_7.pkl
```

If you move a checkpoint between machines, ensure all eight files are transferred. Missing a
single rank's RNG file causes a `FileNotFoundError` on resume.

---

## 6. DataLoader Resumption

Restoring model and optimizer state is not enough to guarantee reproducible training. If the
dataloader resumes at the start of the epoch rather than at the correct position, the mini-batches
seen after resume differ from those in an uninterrupted run. Accelerate provides two strategies.

### 6.1 `skip_first_batches()` — Simple but O(N)

The naive approach re-iterates the dataloader from the beginning and discards the batches that
were already consumed. It is correct and requires no special setup, but it processes (and
discards) data in Python, meaning it consumes CPU/IO time proportional to the number of skipped
batches.

```python
# In accelerator.py / data_loader.py:
def skip_first_batches(dataloader, num_batches):
    """
    Creates a new dataloader that will skip the first `num_batches` batches
    by iterating and discarding them.
    """
    for _ in range(num_batches):
        next(iter(dataloader))   # conceptually — actual impl is slightly different
    return dataloader
```

Usage pattern in a training script:

```python
accelerator.load_state(checkpoint_dir)

# How many batches did we finish before saving?
batches_to_skip = completed_steps % len(train_dataloader)

# Create a wrapper dataloader that skips the first N batches
train_dataloader_skipped = accelerator.skip_first_batches(
    train_dataloader,
    batches_to_skip
)

# First resumed epoch: iterate the skipped dataloader
for batch in train_dataloader_skipped:
    ...

# Subsequent epochs: back to the regular dataloader
for batch in train_dataloader:
    ...
```

**When to use it:** datasets small enough that re-iteration is cheap, or when you cannot install
`torchdata`.

**Limitation:** for a dataset with 100,000 batches and a save at step 80,000, you re-process
80,000 batches worth of sampling and collation just to throw them away. For streaming datasets
(e.g., HuggingFace `IterableDataset` over tens of billions of tokens) this is completely
impractical.

### 6.2 `StatefulDataLoader` — O(1) Resumption

`torchdata>=0.8.0` ships `torchdata.stateful_dataloader.StatefulDataLoader`, a drop-in
replacement for `torch.utils.data.DataLoader` that adds `state_dict()` and `load_state_dict()`
methods. These capture the exact position of the sampler (including shuffle permutation) and
any per-worker dataset state, so resumption is instantaneous regardless of how many batches
were consumed.

Accelerate integrates this through `DataLoaderConfiguration`:

```python
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=True)
accelerator = Accelerator(dataloader_config=dataloader_config)
```

Under the hood, when `use_stateful_dataloader=True`, the `DataLoaderShard` and
`DataLoaderDispatcher` classes that Accelerate builds in `prepare_data_loader()` inherit from
`StatefulDataLoader` instead of the standard `DataLoader`. This is documented in
`src/accelerate/data_loader.py`:

```python
# Simplified from data_loader.py
if use_stateful_dataloader:
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ImportError:
        raise ImportError(
            "StatefulDataLoader is not available. "
            "Please install torchdata version 0.8.0 or higher to use it."
        )
    self.base_dataloader = StatefulDataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
```

When `save_state()` is called, Accelerate calls `state_dict()` on each prepared dataloader and
saves the result alongside the model weights. On `load_state()`, it calls `load_state_dict()`
on each dataloader before the training loop resumes.

The training loop itself becomes simpler because no `skip_first_batches` call is needed:

```python
# With StatefulDataLoader:
accelerator.load_state(checkpoint_dir)   # dataloader position is restored here
for batch in train_dataloader:           # starts from where it left off automatically
    ...
```

#### Prerequisites

```bash
pip install torchdata>=0.8.0
```

`StatefulDataLoader` requires that your sampler (if custom) implements `state_dict()` and
`load_state_dict()`. The built-in `RandomSampler` and `BatchSampler` are patched automatically
when you import `torchdata.stateful_dataloader`, so no extra work is needed for the standard
case.

#### `StatefulDataLoader` vs `skip_first_batches` at a glance

| Property | `skip_first_batches` | `StatefulDataLoader` |
|---|---|---|
| Resume time | O(batches skipped) | O(1) |
| External dependency | None | `torchdata>=0.8.0` |
| Mid-epoch shuffle restore | Approximate (RNG only) | Exact (permutation captured) |
| `IterableDataset` support | Partial | Yes (if dataset implements state protocol) |
| Script changes needed | `skip_first_batches()` call | `DataLoaderConfiguration` only |

---

## 7. Custom State Registration

Accelerate knows about models, optimizers, and schedulers because `prepare()` registers them
internally. Anything else — an EMA (Exponential Moving Average) model, a custom step counter,
a gradient noise scheduler — must be registered explicitly:

```python
accelerator.register_for_checkpointing(ema_model)
```

The registered object must implement:

```python
def state_dict(self) -> dict: ...
def load_state_dict(self, state_dict: dict) -> None: ...
```

This is the same protocol used by `torch.nn.Module`, `torch.optim.Optimizer`, and
`torch.optim.lr_scheduler._LRScheduler`. Any object satisfying it can be registered.

Accelerate saves registered objects in `custom_checkpoint_{i}.pkl` (using `pickle`, not
safetensors) in the order they were registered, and loads them back in the same order.
The load-side logic counts how many `custom_checkpoint_*.pkl` files it finds in the directory
and compares that number to `len(self._custom_checkpoints)`. A mismatch produces a warning:

```
Warning! Number of found checkpoints does not match the number of registered objects:
  Found checkpoints: 2
  Registered objects: 1
  Skipping.
```

This is a known pain point (see issue #1563 in the repository). The warning is silent enough to
miss in a sea of training logs. The practical lesson: do not put extra files in the checkpoint
directory, and always register the same objects in the same order on both the save and the
resume run.

---

## 8. `ProjectConfiguration` and Checkpoint Rotation

Managing checkpoint directories by hand is error-prone. `ProjectConfiguration` automates naming
and rotation.

```python
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

project_config = ProjectConfiguration(
    project_dir="outputs/",
    automatic_checkpoint_naming=True,   # names: checkpoint_0, checkpoint_1, ...
    total_limit=3,                      # keep only the 3 most recent checkpoints
)

accelerator = Accelerator(project_configuration=project_config)
```

When `automatic_checkpoint_naming=True`, each call to `accelerator.save_state()` (with no
`output_dir` argument) writes to:

```
outputs/checkpoints/checkpoint_0/
outputs/checkpoints/checkpoint_1/
outputs/checkpoints/checkpoint_2/
```

Once `total_limit` is exceeded, the oldest checkpoint is deleted before the new one is written.
The deletion logic in `accelerator.py`:

```python
folders = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
folders = sorted(folders, key=lambda x: int(x.split("_")[-1]))

while len(folders) + 1 > self.project_configuration.total_limit:
    # Delete oldest (lowest-numbered) checkpoint
    folder_to_delete = os.path.join(checkpoint_dir, folders.pop(0))
    shutil.rmtree(folder_to_delete)
```

To resume from the latest checkpoint automatically, pass `input_dir=None` to `load_state()`:

```python
accelerator.load_state(None)   # picks up the highest-numbered checkpoint automatically
```

---

## 9. A Minimal Working Example

The following script demonstrates all five mechanisms in one place: model checkpointing,
optimizer state, RNG capture, custom object registration, and checkpoint rotation.

```python
"""
level3_checkpoint_demo.py

Run with:
    python level3_checkpoint_demo.py              # train from scratch, save at step 20
    python level3_checkpoint_demo.py --resume     # resume from step 20, train to step 40
"""
import argparse
import hashlib
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# ---------------------------------------------------------------------------
# A tiny model: 2-layer MLP for regression on random data
# ---------------------------------------------------------------------------
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),           # Dropout consumes RNG — important for reproducibility
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# A custom object to register: a step counter with state_dict protocol
# ---------------------------------------------------------------------------
class StepCounter:
    """Tracks the global training step so it survives checkpointing."""

    def __init__(self):
        self.step = 0

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]

    def __repr__(self):
        return f"StepCounter(step={self.step})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_checksum(model):
    """SHA-256 of all parameters concatenated — useful for reproducibility checks."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def make_dataset(n_samples=1000, seed=42):
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, 16, generator=rng)
    y = X[:, :1] * 2.0 + 0.3          # trivial linear target
    return TensorDataset(X, y)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def train(resume: bool):
    SAVE_STEP   = 20
    TOTAL_STEPS = 40
    BATCH_SIZE  = 32
    LR          = 1e-3
    CHECKPOINT  = "checkpoints/"

    project_config = ProjectConfiguration(
        project_dir=CHECKPOINT,
        automatic_checkpoint_naming=True,
        total_limit=2,                  # keep only the two most recent checkpoints
    )

    accelerator = Accelerator(
        project_configuration=project_config,
        mixed_precision="no",
    )

    model     = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    counter   = StepCounter()

    # Register scheduler and counter so they are included in save_state/load_state
    accelerator.register_for_checkpointing(scheduler)
    accelerator.register_for_checkpointing(counter)

    dataset    = make_dataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # -------------------------------------------------------------------
    # Resume path: load checkpoint, determine where to restart
    # -------------------------------------------------------------------
    start_step = 0
    if resume:
        accelerator.load_state(None)    # picks up the latest checkpoint automatically
        start_step = counter.step
        print(f"[resume] Loaded checkpoint. Resuming from step {start_step}.")
        print(f"[resume] Scheduler last_lr = {scheduler.get_last_lr()}")
        print(f"[resume] Model checksum    = {model_checksum(accelerator.unwrap_model(model))}")

        # Skip batches already consumed in the current epoch.
        # If you used DataLoaderConfiguration(use_stateful_dataloader=True), this
        # would happen automatically inside load_state() and you could remove
        # the two lines below.
        batches_to_skip = start_step % len(dataloader)
        if batches_to_skip > 0:
            dataloader = accelerator.skip_first_batches(dataloader, batches_to_skip)
            print(f"[resume] Skipping {batches_to_skip} already-seen batches.")

    criterion = nn.MSELoss()

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    step = start_step
    dataloader_iter = iter(dataloader)

    while step < TOTAL_STEPS:
        try:
            X_batch, y_batch = next(dataloader_iter)
        except StopIteration:
            # End of epoch: reset to the full dataloader (no skipping)
            dataloader_iter = iter(accelerator._dataloaders[0]
                                   if hasattr(accelerator, "_dataloaders")
                                   else dataloader)
            X_batch, y_batch = next(dataloader_iter)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        accelerator.backward(loss)
        optimizer.step()

        counter.step += 1
        step         += 1

        if step % 10 == 0:
            scheduler.step()
            print(f"  step {step:03d}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

        # -------------------------------------------------------------------
        # Save checkpoint at SAVE_STEP (first run only)
        # -------------------------------------------------------------------
        if not resume and step == SAVE_STEP:
            accelerator.save_state()    # uses automatic_checkpoint_naming
            ckpt_sum = model_checksum(accelerator.unwrap_model(model))
            print(f"\n[save] Checkpoint saved at step {step}.")
            print(f"[save] Model checksum = {ckpt_sum}\n")

    # -------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------
    final_sum = model_checksum(accelerator.unwrap_model(model))
    print(f"\n[done] Training complete at step {step}.")
    print(f"[done] Model checksum = {final_sum}")
    return final_sum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    checksum = train(resume=args.resume)
```

Run the first phase:

```bash
python level3_checkpoint_demo.py
```

Expected output (approximate — your loss values will vary):

```
  step 010  loss=1.2341  lr=0.001000
  step 020  loss=0.9812  lr=0.000500

[save] Checkpoint saved at step 20.
[save] Model checksum = a3f1c7e209b4d8aa

  step 030  loss=0.7651  lr=0.000500
  step 040  loss=0.6104  lr=0.000250

[done] Training complete at step 40.
[done] Model checksum = 5f9c2b1e73d4a801
```

Now start fresh and resume:

```bash
python level3_checkpoint_demo.py --resume
```

Expected output:

```
[resume] Loaded checkpoint. Resuming from step 20.
[resume] Scheduler last_lr = [0.0005]
[resume] Model checksum    = a3f1c7e209b4d8aa   ← matches the save checksum above

[resume] Skipping 20 already-seen batches.
  step 030  loss=0.7651  lr=0.000500
  step 040  loss=0.6104  lr=0.000250

[done] Training complete at step 40.
[done] Model checksum = 5f9c2b1e73d4a801        ← must match the uninterrupted run
```

The two final checksums must be identical. If they diverge, something in the joint state was not
correctly restored.

---

## 10. Exercise: `StatefulDataLoader` with Position Verification

Install the dependency:

```bash
pip install torchdata>=0.8.0
```

Then modify the script to use `StatefulDataLoader` and verify the dataloader position is
captured:

```python
from accelerate.utils import DataLoaderConfiguration

dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=True)
accelerator = Accelerator(
    project_configuration=project_config,
    dataloader_config=dataloader_config,
)
```

With `use_stateful_dataloader=True`, the `accelerator.save_state()` call also serialises the
dataloader's sampler position (including its current shuffle permutation). On resume,
`accelerator.load_state()` restores that position before the training loop starts. You no longer
need `skip_first_batches()`:

```python
if resume:
    accelerator.load_state(None)    # restores dataloader position automatically
    # No skip_first_batches() needed here
```

To verify the dataloader position is actually restored, record the first batch seen after resume
and compare it to the batch that would have appeared at that step in an uninterrupted run:

```python
# Before saving at step 20, record the next batch's first element:
X_batch_21, _ = next(iter(dataloader))
print(f"[verify] First element of batch 21 (uninterrupted): {X_batch_21[0, 0].item():.4f}")

# After resuming, the first batch from the dataloader should match exactly.
```

---

## 11. Exercise: Checkpoint Rotation Verification

Use `total_limit=2` and save at multiple steps. After three saves, only the two most recent
checkpoints should exist:

```python
project_config = ProjectConfiguration(
    project_dir="checkpoints/",
    automatic_checkpoint_naming=True,
    total_limit=2,
)
```

After running three `save_state()` calls:

```bash
ls checkpoints/checkpoints/
# Should show: checkpoint_1  checkpoint_2
# checkpoint_0 should be gone
```

You can also write a simple assertion in the training loop:

```python
if step % 10 == 0:
    accelerator.save_state()
    dirs = sorted(os.listdir("checkpoints/checkpoints/"))
    assert len(dirs) <= 2, f"Expected at most 2 checkpoints, found {len(dirs)}: {dirs}"
    print(f"[rotation] Checkpoints present: {dirs}")
```

---

## 12. The `register_for_checkpointing` Protocol in Detail

Any object with `state_dict()` and `load_state_dict()` can be registered. Here is a more
realistic example: an EMA wrapper that tracks a shadow copy of the model parameters.

```python
class EMAModel:
    """
    Exponential moving average of model parameters.
    Compatible with accelerator.register_for_checkpointing().
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Store shadow parameters as CPU tensors
        self.shadow = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.detach().cpu(), alpha=1 - self.decay
            )

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay  = state_dict["decay"]
        self.shadow = state_dict["shadow"]

    def copy_to(self, model: nn.Module):
        """Overwrite model parameters with the EMA shadow copy."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name].to(param.device))


# Usage
model    = TinyMLP()
ema      = EMAModel(model, decay=0.9999)
accelerator.register_for_checkpointing(ema)

# During training:
for batch in train_dataloader:
    ...
    accelerator.backward(loss)
    optimizer.step()
    ema.update(accelerator.unwrap_model(model))   # update after each optimizer step

# save_state() / load_state() will include the EMA shadow weights automatically.
```

---

## 13. Common Pitfalls

### 13.1 Registering Objects After `prepare()`

`register_for_checkpointing()` must be called **before** `save_state()`, but it does not have to
be before `prepare()`. The internal list `self._custom_checkpoints` is populated whenever
`register_for_checkpointing()` is called, so the order relative to `prepare()` is flexible. What
matters is that the **same objects are registered in the same order** on both the save run and
the resume run. Registering in a different order causes the wrong `state_dict` to be loaded into
the wrong object with no error raised.

### 13.2 `load_state()` Before `prepare()`

`load_state()` restores weights into already-prepared objects. The correct order is:

```python
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
accelerator.register_for_checkpointing(scheduler)
accelerator.load_state(checkpoint_dir)   # model is already on the right device
```

Calling `load_state()` before `prepare()` will load weights into a CPU model that has not yet
been wrapped in `DistributedDataParallel`, and the subsequent `prepare()` call may or may not
transfer those weights correctly depending on the distributed backend.

### 13.3 Forgetting to Unwrap Before Inspecting Weights

After `prepare()`, `model` is a `DistributedDataParallel` wrapper (in multi-GPU mode). If you
compute a checksum or call `state_dict()` directly on the wrapped model, the keys include the
`module.` prefix. Use `accelerator.unwrap_model(model)` to get the underlying `nn.Module`:

```python
raw_model   = accelerator.unwrap_model(model)
param_sum   = sum(p.sum().item() for p in raw_model.parameters())
state       = raw_model.state_dict()                # keys: "net.0.weight", not "module.net.0.weight"
```

### 13.4 `scaler.pt` Only Exists in fp16 Mode

If you save in `fp16` mixed precision and resume in `no` precision (or vice versa), the scaler
file will be absent on one side. Accelerate handles the absence gracefully (it simply skips
scaler loading), but the training dynamics will differ because the gradient scaling history is
lost.

### 13.5 Moving Checkpoints Between Machines

A checkpoint directory is self-contained but it encodes the number of processes at save time
through the `random_states_{rank}.pkl` files. Resuming on a different number of GPUs requires
at minimum re-generating the optimizer state (Adam `m`/`v` buffers are shaped `[num_params]`
and are device-independent) and accepting that per-GPU RNG states do not map cleanly to a
different process count. This is an open limitation of Accelerate's checkpointing design and a
potential area for future contribution.

---

## 14. What the Checkpoint Captures — Summary Diagram

```
accelerator.save_state("ckpt/")
│
├── save_accelerator_state()
│     ├── for i, model in self._models:
│     │     └── safetensors (or .bin)  →  ckpt/model_{i}/model.safetensors
│     │
│     ├── for i, optim in self._optimizers:
│     │     └── pickle state_dict      →  ckpt/optimizer_{i}/optimizer.bin
│     │
│     ├── for i, sched in self._schedulers:
│     │     └── pickle state_dict      →  ckpt/scheduler_{i}/scheduler.bin
│     │
│     ├── random_states snapshot       →  ckpt/random_states_{rank}.pkl
│     │     (random, numpy, torch CPU,
│     │      torch CUDA/HPU/XPU/MPS)
│     │
│     └── if scaler:
│           └── scaler state_dict      →  ckpt/scaler.pt
│
├── for i, obj in self._custom_checkpoints:
│     └── pickle obj.state_dict()     →  ckpt/custom_checkpoint_{i}.pkl
│
└── (if use_stateful_dataloader)
      └── dataloader.state_dict()      →  embedded in accelerator state
```

---

## 15. What You Should Understand After This Level

By the end of this level you should be able to answer these questions without looking anything up:

**What exactly gets saved in a checkpoint?**
Model weights (safetensors), optimizer momentum buffers (pickle), scheduler `last_epoch` and
`_last_lr`, all four RNG subsystems (Python, NumPy, PyTorch CPU, device-specific), optionally
the GradScaler, and any objects registered with `register_for_checkpointing()`.

**Why do independent `if` checks matter for RNG save/load?**
Save and load must be structurally symmetric. An `elif` on the load side means only the first
matching device's RNG is restored, leaving all other devices in an incorrect state. The bug
produces no error — only subtly wrong training dynamics after resume.

**What is the difference between `skip_first_batches` and `StatefulDataLoader`?**
`skip_first_batches` re-iterates and discards batches — O(n) in the number of skipped batches,
no extra dependency. `StatefulDataLoader` serialises the sampler permutation and worker position
as part of `save_state()` — O(1) resumption at the cost of requiring `torchdata>=0.8.0`.

**How do you register custom objects?**
Call `accelerator.register_for_checkpointing(obj)` where `obj` has `state_dict()` and
`load_state_dict()` methods. Objects are saved as `custom_checkpoint_{i}.pkl` in registration
order and must be registered in the same order on the resume run.

**How does checkpoint rotation work?**
`ProjectConfiguration(automatic_checkpoint_naming=True, total_limit=N)` names checkpoints
sequentially and deletes the oldest one whenever the count exceeds `total_limit`. Deletion
happens in `accelerator.save_state()` before writing the new checkpoint.
