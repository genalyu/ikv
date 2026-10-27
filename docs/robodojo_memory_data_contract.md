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
