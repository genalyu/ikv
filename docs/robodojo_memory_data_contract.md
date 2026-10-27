# RoboDojo Memory data contract

The six-task campaign must preserve episode-level splits across the scorer and both policies.
A dataset view can contain original metadata and immutable data symlinks plus
`episode_selection.json` with a nonempty list of `episode_indices`; only those episodes are
sampled. The valid-segment cache includes this selection in its identity. Validation must
be enabled explicitly in an aggregate task (`validation.enabled`, `validation.interval`).
Previously trained all-task weights are not clean initializations if they saw these held-out
episodes. Use a released base checkpoint for formal comparisons.

RoboDojo JPEGs use a marker-aware storage convention. Set `runtime.xpolicylab_root` to the
pinned XPolicyLab package and use its `decode_image_bit`; generic PIL decoding reverses
legacy RGB. Conversion fingerprints include the decoder digest. Existing conversions and
all derived visual features must be regenerated in a fresh output directory.

The latent encoder now selects source frames explicitly. Integer source/target FPS ratios
are required; an ffmpeg FPS filter followed by guessed frame IDs is forbidden. New payloads
record `frame_sampling_contract=explicit-source-stride-v1`. For the Memory campaign use
`preprocessing.rgb_resize_mode=torch_bilinear` to match serving's float bilinear interpolation
with align_corners=False; `area` remains available for other recorded recipes.

`fit_eef20_bounds` fits absolute EEF20 bounds using TRAIN data only: physical [-1,1] rotation,
[0,1] grippers, and training-workspace position bounds with a small margin. It never clips.
The legacy q01/q99 field names carry bounds, not empirical percentiles; the version field
makes that distinction explicit. Bundle the exact stats and split hash with every run and
serving config. Changing stats requires a new run, not hot-changing a checkpoint's contract.

RGB-only task overlays preserve an explicitly chosen scoring backend rather than silently
resetting it to formula. This does not implement a new scorer backend: supported backend,
causal training integration and streaming inference must be independently validated.

Unit tests and loss curves do not establish task success. Before long training, require real
four-GPU updates, full optimizer/scheduler/RNG checkpoint restore, physical action round trips,
train/serve observation parity, and closed-loop development evaluation with original rewards.
The swap_blocks button release issue is an evaluation environment confound and must not be
fixed by lowering the success threshold or imputing wins.


## Frozen scorer integration

`training.memory.prepare_plans` binds a checkpoint hash and pinned Qwen encoder,
encodes corrected RGB at actual VAE anchors, and replays a causal eviction policy.
Scorer frame/camera/row/column scores are explicitly permuted into native
frame/row/camera/column KV order. For a native two-frame group starting at f, both video and action phases use
only scorer inputs from frames before f. The group drives toward future video
frames f and f+1, whose real RGB is not yet available to the scorer. Evicted native KV cannot re-enter. Budgets count
current plus history (3584 RGB, 512 action); no tactile tokens are admitted.

`training.memory.train` loads these plans with B=1, complete trajectories and
fixed native chunk size 2. N0 uses its existing masked IKV path. The LingBot adapter imports an
explicit upstream source directory, keeps its architecture/denoising loss, uses
the audited episode-filtered data loader, and adds durable DCP state and held-out
validation. The original baseline remains a separately named control. Scorer
failure or a missing/stale plan raises; there is no formula fallback.

The offline scorer sees VAE-anchor RGB, which is temporally sparser than its SFT
corpus. Calibration on this exact representation and streaming serving parity
remain required acceptance gates. Integration smoke tests using an older head
are not evidence of the final scorer's performance, task success or SOTA.

The DCP loader now respects uninitialized Adam state for unused branches. It
uses checkpoint metadata to omit only entirely absent parameter states; missing
fields of an active state still error. This avoids silently advancing Adam's
bias-correction step for a branch before its first actual gradient.


The first twelve action tokens are seeded from measured initial
observation.state, matching real-observation grounding, rather than repeating
the first future command/pose target. This is opt-in for both Memory controls.
The complete action suffix is unchanged. CPU loaders run with zero workers in
the audited recipe so an exact resume does not depend on opaque worker RNG.

Both trainers reject nonfinite losses collectively before backward and reject
nonfinite gradient norms before an optimizer update. Rank-zero metrics retain
full floating-point precision for save/resume comparison. LingBot exports
inference weights one parameter at a time into bounded safetensors shards,
avoiding a full-model host-memory gather under container memory limits.
Checkpoint completion is invalidated before rewriting any part of a checkpoint.

training.memory.timebase defines the exact 25 Hz control / 30 Hz action /
10 Hz RGB mapping. Control actions select ceil(k*30/25) from native predictions,
while grounding RGB arrives at floor(m*25/10). Test the actual deployed adapters
as well as these arithmetic helpers; synthetic/instrumented adapters do not
establish simulator success. Online scorer integration and measured cache
selection parity remain separate from offline-plan validation.


## Native frozen-scorer serving

The optional Memory serving entry is training.memory.serve. It accepts an
external JSON naming model, complete checkpoint directory, base_model, bundle,
save_root, and (for LingBot) source_root. A scorer checkpoint also requires its
binding and scorer_endpoint. The training memory_manifest.json owns camera
order, action dimensions, normalization and scorer hash; incompatible or partial
checkpoints fail before model loading. The bundle links the trained transformer
and the recorded base VAE/tokenizer/text encoder.

Run training.memory.online_scorer in the Qwen-compatible environment with
--binding and --port; it binds localhost by default. Policy inference uses
ScorerClient in its own environment. Only measured RGB VAE anchors enter the
feature bank. Both clients pin the checkpoint hash and protocol. Prefix queries
exclude current/future target frames, even if the feature bank has already
received a later observed frame. Contextual scores are recalculated per prefix.

FrozenMemoryRetention maps actual RoPE coordinates to native RGB/action IDs,
then uses the same eviction function as offline plans. N0 attaches it to the
existing shared MoT cache. LingBotCache overrides only slot allocation and keeps
upstream attention arithmetic; temporary diffusion passes and failed model
forwards restore every layer. A failed policy request requires episode reset,
because the upstream LingBot VAE does not expose a transaction API.

LingBot's websocket state payload is raw joint14, matching its original
preprocess_action. That function expands it once into the model's thirty
channels. Sending a pre-expanded thirty-channel state would map the controls
twice and corrupt grounding. N0 receives observed EEF20 in its native contract.

Configuration validation, CPU attention checks, full-model recorded-observation
replay, XPolicyLab encoded-client debug, and simulator evaluation are distinct
evidence. Record which passed. A replay must exercise actual budget overflow,
compare native selected IDs to the offline plan, check streaming VAE parity,
and check finite native actions. It does not measure task success.


The Memory serving recipe records cold_seed_mode=free for both controls.
The first action frame is denoised toward its measured-state training target
and skipped by execution. The pinned LingBot server has a separate optional
serving patch for this mode; its legacy zero-clamp default is unchanged.
This shared boundary choice needs matched-control closed-loop evaluation and
must not be attributed to scorer retention gains.
