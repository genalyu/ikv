# VPP2 and HWM-0 integration patches

These patches target the exact upstream revisions in manifest.json.
In a clean upstream checkout at that revision, use git apply --check followed
by git apply with the corresponding patch. Install this IKV core checkout
(or put it on PYTHONPATH) to supply ikv_runtime.

The patches implement window_refresh and incremental_reuse histories.
Native baseline is history_mode=off. Formula examples use the Stage 1
heuristic; learned scorer examples keep an independent Qwen encoder.
HWM formula requires lower_only; learned scoring also supports interactive
Planner feedback. Changing history modes requires episode reset.

Tests cover transactional archives, tiny native models and scorer contracts.
These patches do not implement two-host pipeline parallel deployment and
do not imply full-checkpoint or RoboDojo task-score validation.
Machine-specific launchers, checkpoints and evaluation artifacts are excluded.
