# RoboDojo RGB-only N0-TWAM IKV

RoboDojo simulation uses the explicit robodojo_hdf5 adapter. Set rgb_only=true,
robot.tactile={}, action_labels=next_observation and runtime.robodojo_reader_root
to a checkout of the RoboDojo-aware OpenWAM reader. The reader applies the released
dual-X5 environment-to-arm-base transform and column rot6d EEF20 packing.
Pin that reader revision with the run. Supervision is next achieved EEF state,
not an assertion that commanded EEF targets were recorded. JPEG decoding uses
PIL RGB. Native 25 Hz observations/actions are causally resampled to the existing
30 Hz post-training clock, with alignment metadata recorded. No tactile is synthesized.

Each of the 35 task overlays retains its own instruction, RGB latents, semantic
index provenance, and action normalization. An aggregate overlay's prepared_tasks
lists those overlays; its pool uses task-name symlinks. Missing datasets abort
initialization rather than being silently omitted. Semantic provenance is checked
against the corresponding task, including its prompt hash.

The RGB-only retention recipe uses formula scoring, video/action/tactile budgets
3584/512/0, full RGB support and no motion. Persistence, class recency and (with a
semantic encoder) task relevance have unit weights; contact, query usage, legacy
visual similarity and legacy time score weights are zero. Action KV follows FIFO.
The new Qwen Sys2 inference mode remains opt-in and is not used for this training.

The initial experimental schedule is 10,000 optimizer updates, per-GPU batch 1,
four GPUs and accumulation 4 (effective batch 16), learning rate 1e-5, 100 warmup
updates, cosine decay and checkpoint interval 500. These are proposed RoboDojo
adaptation settings, not an official N0-TWAM RoboDojo reproduction recipe.
Full model four-GPU forward/backward and checkpoint validation must precede the
production run. Training restores optimizer state only from completed manifests.
