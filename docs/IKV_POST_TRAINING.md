# Post-train motion patches and IKV with the N0-TWAM recipe

Start from the released **base** MoT checkpoint. Use the existing `train.py`,
LeRobot dataset, per-task action normalization, condition corruption, diffusion
noise schedule, teacher forcing, video/action/tactile Flow Matching losses,
AdamW, gradient accumulation, activation checkpointing and FSDP. There is no
separate rollout objective, RL, distillation, LoRA or learned retention scorer.
The original trainer updates its normal trainable transformer parameters; these
switches change what context they learn to use, not which parameter names train.
Frozen preprocessing features never become model input embeddings.

## Multimodal retention version 2

The posttrain configuration now explicitly selects version 2 when IKV is enabled.
Baseline with IKV disabled is unchanged. Legacy configurations without a version
remain version 1; loading a checkpoint compares the complete retention config and
does not silently upgrade it.

Version 2 divides the total capacity into video/action/tactile budgets. Configure
`kv_retention.video_capacity`, `action_capacity`, and `tactile_capacity`; their
sum must equal `ikv_train_capacity`. The CLI override is
`--ikv-modality-capacities VIDEO ACTION TACTILE`. The posttrain defaults are
2816/512/768. These are configurable starting budgets, not measured optimums.

- Motion selects visual patches only; all modalities still participate in the
  original shared attention. KV keeps RoPE, but the version-2 importance index
  stores no spatial coordinates, camera IDs, sensor IDs, or action vectors.
- The current post-training recipe sets all three query-usage weights to zero
  for both training and serving. Visual score combines persistence, contact
  duration, current DINO relevance and recency. Tactile score uses recency.
  Action retention first prefers rows with surviving same-frame visual or active
  tactile evidence, then recency. Action subframe timestamps map back to their
  parent frame for this association. These scores select context; model
  attention remains trainable. Legacy checkpoints may still use query usage.
- Separate modality budgets prevent a high score in one modality from taking
  another's budget. Old candidates are evicted in stable ascending score order;
  incoming keys must fit their modality budget. Same-score ties evict oldest
  time, then oldest insertion. There is no random remainder eviction in v2.
- DINO persistence measures observed support, not cache residence. Full-grid
  features before motion are required when `persistence_weight > 0`, even if
  some/all patches were filtered. One group/time counts once, observation gaps
  do not count, and content history is bounded by `content_capacity`. Similarity
  is a heuristic content association, not guaranteed physical object identity.
- `MultimodalKVRetention.tactile_scorer` is an optional detached scoring hook
  accepting times and NeoForce vectors. It is unset in this release; no Qwen
  calls or judge training are introduced. Any future implementation must also
  version and validate its train/serve configuration.

For v2, `ikv_index_root_name` full-grid sidecars are used in both motion and dense
training. Supply `dino_features[F, spatial, D]` and optionally
`frame_neoforce_features[F,D]` (zero without contact). Data loading preserves the
full-grid DINO tensor as `dense_dino_features`, independently of sparse support.
These features are detached; causal phases update content history only after the
phase has been evaluated. Set `persistence_weight=0` to explicitly disable P
when full-grid DINO is unavailable.

Serving v2 accepts `contact_index={neoforce,response}` covering existing tactile
tokens in sensor/frame/patch order. Positive response gates descriptors; their
frame mean is broadcast to every visual token of that frame. An independent
contact flag prevents vector cancellation from being mistaken for no contact.
No visual-row matching is performed. Legacy `contact_pairs/visual_rows` is
rejected in v2. Optional online NeoForce uses the same frame-level route; its
existing force-input schema and encoder weights remain unchanged.

The version-2 CUDA smoke is:
`OMP_NUM_THREADS=2 python tests/smoke_ikv_training_cuda.py --retention-version 2`.
It checks tiny random models, not task performance or full-model memory fit.

## Independent switches

First complete [POST_TRAINING.md](POST_TRAINING.md): convert demonstrations,
encode latents, create a task pool and normalization statistics, and edit the
paths/camera keys/prompt in `n0_twam/configs/twam_posttrain_cfg.py`. Raw simulation
HDF5 is not a ready-to-train sample; it must contain or be converted to the
original aligned absolute action targets and observations.

```bash
# Original recipe
NGPU=8 bash run_posttrain.sh --base-checkpoint /path/to/n0-twam-base --no-motion --no-ikv
# Motion only
NGPU=8 bash run_posttrain.sh --base-checkpoint /path/to/n0-twam-base --motion --no-ikv
# IKV only, full RGB tokens
NGPU=8 bash run_posttrain.sh --base-checkpoint /path/to/n0-twam-base --no-motion --ikv --ikv-capacity 4096
# Both
NGPU=8 bash run_posttrain.sh --base-checkpoint /path/to/n0-twam-base --motion --ikv --ikv-capacity 4096
```

`--base-checkpoint` names a directory containing `transformer/` and initializes
model weights through the original loader. Set `_BASE_MODEL` / `_EMPTY_EMB`
separately for the VAE/text assets. CLI switches override `_USE_MOTION`,
`_USE_IKV`, `_IKV_CAPACITY`; both features default off in the posttrain recipe.
For independent per-task experiments, make a separate pool, norm file and output
directory for each task, and start each from the same base checkpoint.
GPU count above follows the existing launch recipe, not a measured minimum.

## What changes inside a training sample

With IKV off, attention uses the original packed sequence mask and random local
window. It does **not** have an unlimited online cache: its accessible context is
bounded by the sampled sequence and attention window, and there is no streaming
slot eviction during that forward pass.

With IKV on, the same teacher-forced sequence uses the original causal
video/tactile and action phases with separate noisy and condition branches.
A noisy token cannot read its own phase's condition tokens. The detached
retention policy first selects the historical condition tokens visible to each
branch. The normal full-sequence MoT forward then runs once with that mask,
so autograd trains the same attention parameters without a recurrent KV graph.
There is no additional distance window in this IKV mask: a retained old token
remains visible regardless of age. For masked IKV training, each sample draws a
token capacity K uniformly between a configured fixed minimum (half the
maximum by default) and the configured maximum. This draw is independent of
the sample's future tokens. Version 2 preserves separate video/action/tactile
budgets: it draws total K and distributes slots proportionally across each
modality's configured capacity. Inference uses the
configured full capacity. Baseline instead randomly draws a time window of
4-64 phase IDs; these are different units and are not numerically equivalent.

- Capacity is measured in valid tokens per sample, per layer. Version 2 enforces
  separate modality budgets; legacy version 1 shares one budget. If a
  sampled budget cannot fit the current phase, training fails explicitly;
  raise the task's `ikv_min_capacity` for that robot/task. No token is silently
  dropped to fit K. Low scores are evicted first, with
  older tokens breaking equal-score ties in version 2.
- Later action loss can still backpropagate through earlier condition tokens
  because they participate in the single differentiable MoT forward. Only hard
  retention selection and index features are detached.
- The selected version's serving retention policy is reused with query-usage
  weights zero. Current clean DINO must not affect which history a simultaneous
  noisy target sees; noisy selection uses the previously committed anchor.
- Each sample/forward starts an empty history, including across packed batch
  members. Nothing carries over to a different episode or optimizer step.
- This remains teacher forcing. It does not train through multi-step diffusion
  sampling or imagined online trajectories. The discrete IKV rule has no learned
  weights; training adapts the transformer to its selected context.

For motion training, condition patches come from the observed frame's sidecar.
Prediction addresses come from the last observed frame of the preceding chunk,
carried to the current chunk's original temporal/spatial coordinates. Neither
future target patch selection nor future DINO is supplied to the prediction.
The first chunk has no previous support, so its video prediction loss is masked;
its observed condition patches still supervise subsequent action predictions.
NeoForce is not fabricated for predicted patches. A previous tactile-only address
without visual metadata stays in the observed branch but is excluded from visual
prediction support. Padding and both branches' time/validity layouts are separate.

Example: an early observation shows which socket to use, then several chunks
pass before insertion. The later action still uses the original supervised target
and action loss. If IKV retained the early observation, gradients can teach the
transformer to read it. A sample cropped entirely after that observation cannot
teach this dependency. Include cue and decision in the same training sequence;
choose a capacity and sequence length that actually exercise retention/eviction.

**Memory limit:** the single forward eliminates the recurrent per-phase KV
autograd graph, but model activations still scale with the full training sample.
A small visible history does not imply a small full-sequence training footprint.
Measure the longest sample before starting full training. W&B records the
mean sampled K under `ikv/mean_capacity` for each optimizer update.

## Index data

Motion modes use the existing canonical `rgb_motion` sidecars and loader checks
(camera order, WAN grid, frame alignment, valid masks, DINO/NeoForce). See the
existing motion preprocessing scripts. The sidecar width must cover the complete
selected support; the posttrain config pads to the full configured camera grid,
matching the RGB server. Do not silently truncate moving patches. Dense WAN encoding is
unchanged; sparse selection starts at transformer patches.

For dense IKV-only training, `cfg.ikv_index_root_name = None` means semantic
features are unavailable (their scores are zero); time still operates (action repetition is legacy version 1 only). To train with semantic retention, set an explicit relative root,
e.g. `ikv_index`, and supply one file per encoded segment:

```
<dataset>/<ikv_index_root_name>/chunk-XXX/episode_XXXXXX_START_END.pth
```

Each `torch.save` dictionary must contain:

- `camera_keys`: exact ordered RGB camera keys.
- `patch_size`: transformer patch size, currently temporal size 1.
- `spatial_grid_shape`: height/width of the concatenated camera patch grid.
- `frame_ids`: the original encoded bundle's raw frame ID list, before cropping.
- Optional `dino_features`, `neoforce_features`: finite `[F, Hpatch*Wpatch, D]`
  tensors, one entry per encoded latent token, before cropping. An absent feature
  has width zero. Index tensors receive exactly the video sample's temporal crop.

Use the same feature producer and retention weights in training and serving.
A malformed or misaligned explicit sidecar is an error, not a silent fallback.

## Serving and verification

Checkpoints record the switches, capacity and retention configuration in
`train_meta.json`. Configure `posttrain_server` to match the training switches
(CLI training overrides do not edit the config file). An IKV-trained checkpoint
requires global retention and the same capacity/weights; startup checks reject
mismatches. Motion + global is supported: selected sparse metadata drives the
same policy, and padding consumes no cache slots. Dense DINO re-encoding is gated
off for this combination because its index comes from the motion path.

Regression tests check no-eviction output/gradient parity with the original
attention, checkpoint recomputation, gradients to old KV, causal support,
no target leakage into retention, all four training/loss/backward combinations,
and sparse global serving. These checks establish implementation behavior, not
NeoSim task success or learned long-term memory. Run actual per-task posttraining
and closed-loop held-out evaluations before claiming either.

Reproduce the checks from the repository root:

```bash
OMP_NUM_THREADS=2 python -m pytest tests -q
NCCL_DEBUG=WARN OMP_NUM_THREADS=2 python tests/smoke_ikv_training_cuda.py
```

The CUDA smoke uses a two-layer random MoT on one GPU, the real FSDP wrapper and
activation checkpointing, two accumulated microbatches per update, and two
updates in every switch combination. It checks losses and every parameter's
gradient against an unsharded reference, with BF16 tolerances. This is not a
multi-GPU communication test or a full-size checkpoint memory benchmark.


## 可复用真机 / 仿真任务入口

任务数据分析、LeRobot 转换、1/2/8 卡等效 batch、四种消融及完整恢复见 [任务训练指南](TASK_PIPELINE.md)。


### Reusing exact compact training metadata

Compact masked training can construct its `(query, retained_keys)` groups during
retention replay, without materializing the two `[phase, token]` support tables.
The dense support-plan path remains available as the regression reference. Key
indices retain their original sorted order; capacity sampling, policy decisions,
losses and gradients are unchanged. The packed query/key indices and cumulative
lengths are built once per trajectory and shared across layers and checkpoint
recomputation. Only metadata is reused; Q/K/V tensors and their gradients remain
specific to each layer.

For version-2 persistence, replay submits only previously unseen dense DINO
frames, in chronological order. `ContentHistory.clock` already rejects duplicate
frames, so repeatedly gathering and scanning the growing historical prefix is
unnecessary. Greedy prototype matching and observation durations are unchanged.

Training loaders pin host tensors by default (`pin_memory=False` disables this),
and device transfers use the same CUDA stream with `non_blocking=True`. Worker
lifetime, sample order and RNG handling are unchanged. Training metric reductions
use two vector collectives instead of one mean and one maximum per metric. Each
reduction receives its own copy: a mean reduction must not overwrite the local
values before computing the maximum. This affects logged maxima, not the loss
used for optimization.

Validation: CPU tests compare direct groups with the dense plan (including padded
and nonconsecutive phases), retained history, full outputs/gradients, and two-rank
metric reduction. `tests/smoke_compact_preparation_cuda.py` compares both compact
paths on a small checkpointed CUDA model, verifies one shared pack, all parameter
gradients, and an AdamW update. These checks do not establish a full-size FSDP
throughput or peak-memory result; measure those before changing a production run.


### Reducing host synchronization and nested backward overhead

Content-history matching retains the original GPU dot products and chronological
feature order. It transfers each score vector once for the sequential decision,
instead of separately synchronizing on its argmax and selected score. Frame
spans and slot timestamps are copied once per frame. A frame-local timestamp
list preserves oldest-slot eviction, first-index ties, and exclusion of slots
already seen in that frame; no extra persistent state is introduced. Duration
and timestamp updates for multiple distinct slots are batched once per frame,
with exactly one duration addition per slot.

Bounded compact attention recomputes each packed chunk as before, but invokes
the matching ATen Flash backward directly instead of constructing a nested
autograd graph and calling `autograd.grad`. Q/K/V gradient paths remain intact,
and duplicated K/V gradients still accumulate in FP32. This does not retain
additional full-sequence activations. The CUDA smoke test compares against native
packed-attention autograd at several chunk budgets, including padded tokens.

Component timings on a shared GPU are diagnostic only. Full-trajectory FSDP
forward, backward and optimizer-step timing must still be measured before
claiming a production speedup or changing the running training process.
