# N0-TWAM baseline and IKV post-training

Only baseline and ikv modes are supported. Both consume the full native RGB latent
grid. Motion/motion_ikv modes, RGB-D motion detection, adjacent-frame selection,
motion sidecars, online production, and sparse-motion loss/reconstruction are removed.

## Shared objective

The observation encoder, temporal grid, teacher forcing, noise/LR schedules,
trainable parameters, action/tactile targets and loss weights are shared. IKV
changes historical condition visibility only. It does not discard current RGB patches.
DINOv3 visual/tactile features remain independent index metadata. Contact duration
and retention formulas are unchanged. See SCREENSHOT_HEURISTIC.md and
TACTILE_SEMANTIC_SIDECARS.md.

## Task CLI

    python script/task_pipeline.py prepare --task /path/to/task.json --stage all --device cuda
    python script/task_pipeline.py check --task /path/to/task.json --mode baseline
    python script/task_pipeline.py check --task /path/to/task.json --mode ikv
    python script/task_pipeline.py train --task /path/to/task.json --mode baseline --gpus 8
    python script/task_pipeline.py train --task /path/to/task.json --mode ikv --gpus 8

The task training object owns steps, save interval, LR and accumulation. Full
checkpoint resume preserves optimizer, scheduler, rank RNG and data position.
Do not shorten num_steps for timing because it participates in the LR schedule.

## Existing data and configuration

ikv_index remains the full-grid DINO/contact sidecar. Feature extraction writes
it directly and no longer generates or requires rgb_motion artifacts. Existing
complete dense indexes remain reusable after provenance/alignment validation.
Historical on-disk motion artifacts are not read or deleted.

Removed motion options/modes, explicit enabled legacy flags and incoming motion
payloads are rejected rather than silently interpreted as dense RGB. Sparse
constructor arguments are removed. Dense base/baseline/IKV parameter names and
shapes are unchanged; sparse-motion checkpoints are outside this input contract.

## Within-update workload grouping

IKV_GROUP_WITHIN_UPDATE=1 uses the production WithinUpdateDistributedSampler for
distributed singleton IKV. It reorders samples inside the original effective batch
only, preserves each update's example multiset, and leaves partial epoch windows
untouched. Seeded shuffling, padding, complete corpus, objective and retention
formulas stay unchanged. The sampler layout is recorded in full checkpoint
metadata and resume rejects a mismatched layout.


### Absolute EEF action layout and normalized loss

For absolute EEF rotation-6D tasks, the resolved training and serving configs
use physical bounds [-1, 1] for the six rotation entries of each active arm.
This also upgrades old percentile pool statistics in memory. Other dimensions
retain the task's fitted bounds. Positions and grippers are not silently clipped.
A single-arm trajectory supervises 10 packed source columns, even in a
20-column Parquet file. Dual-arm trajectories supervise all 20. The same mask
is applied to the measured cold seed and subsequent actions. Tensor losses now
average over the actual active channels. The effective bounds and active
channel layout are part of the checkpoint resume contract. Use a new run
when either differs; compare task space metrics separately from normalized MSE.
