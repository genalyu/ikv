# Post-train motion patches and IKV with the N0-TWAM recipe

Start from the released **base** MoT checkpoint. Use the existing `train.py`,
LeRobot dataset, per-task action normalization, condition corruption, diffusion
noise schedule, teacher forcing, video/action/tactile Flow Matching losses,
AdamW, gradient accumulation, activation checkpointing and FSDP. There is no
separate rollout objective, RL, distillation, LoRA or learned retention scorer.
The original trainer updates its normal trainable transformer parameters; these
switches change what context they learn to use, not which parameter names train.
Frozen preprocessing features never become model input embeddings.

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

With IKV on, the same teacher-forced sequence is evaluated in its original causal
phases: video/tactile phase, then action phase, for each sampled chunk. Each phase
has separate noisy and condition branches. A noisy token cannot read its own
phase's condition tokens; both branches can read allowed earlier condition KV.
The existing global retention policy chooses the earlier tokens visible to each
branch. Its capacity replaces the old distance window for history, so retained
old evidence can remain visible after many intervening chunks.

- Capacity is shared across modalities, measured in valid tokens **per sample,
  per layer**. The largest incoming phase must fit, or training fails explicitly.
- Clean/condition KV is kept with its autograd graph. Later action loss can train
  attention to earlier evidence. Only hard selection, index features and usage
  statistics are detached. No detached replay cache is substituted.
- The same global top-k protection and seeded random eviction policy is reused.
  Query usage is measured from condition queries and averaged across layers.
  Current clean DINO must not affect which history a simultaneous noisy target
  sees; noisy selection uses the previously committed observation anchor.
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

**Memory limit:** bounded visible historical KV does not bound the entire training
VRAM footprint. Autograd retains dependencies on earlier phases, including paths
through evicted tokens. Long sequences require measured memory tuning. This code
does not claim full-model training fits a single A100 at any fixed sequence size.

## Index data

Motion modes use the existing canonical `rgb_motion` sidecars and loader checks
(camera order, WAN grid, frame alignment, valid masks, DINO/NeoForce). See the
existing motion preprocessing scripts. The sidecar width must cover the complete
selected support; the posttrain config pads to the full configured camera grid,
matching the RGB server. Do not silently truncate moving patches. Dense WAN encoding is
unchanged; sparse selection starts at transformer patches.

For dense IKV-only training, `cfg.ikv_index_root_name = None` means semantic
features are unavailable (their scores are zero); time/query/action repetition
still operate. To train with semantic retention, set an explicit relative root,
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
