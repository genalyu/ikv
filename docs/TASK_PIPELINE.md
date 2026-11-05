# Native RGB task pipeline

Supported modes are baseline and ikv; both use full RGB observations.
See IKV_POST_TRAINING.md and ROBODOJO_RGB_TRAINING.md.

Task JSON specifies source format/path, prompt, robot camera/tactile mapping,
action representation, model paths and training recipe. Combined child tasks
retain their independent topology, action channels and normalization.

    python script/task_pipeline.py prepare --task /path/to/task.json --stage convert
    python script/task_pipeline.py prepare --task /path/to/task.json --stage rgb --device cuda
    python script/task_pipeline.py prepare --task /path/to/task.json --stage tactile --device cuda
    python script/task_pipeline.py prepare --task /path/to/task.json --stage features --device cuda
    python script/task_pipeline.py prepare --task /path/to/task.json --stage pool
    python script/task_pipeline.py check --task /path/to/task.json --mode ikv
    python script/task_pipeline.py train --task /path/to/task.json --mode ikv --gpus 8 --dry-run

RGB-only tasks omit tactile preparation. Features are full-grid DINOv3 and configured
tactile/contact metadata under ikv_index; motion sidecars are not required.
Source frames, causal VAE anchors, semantic provenance, task identity and complete
corpus counts must pass validation. Missing tasks must not be silently filtered.

Use baseline to train N0-TWAM with ordinary history visibility. The model,
optimizer, loss, noise and temporal grid recipe is shared. Effective batch is
per-GPU batch times GPU count times accumulation; keep the requested recipe.

Resume via train --resume /path/to/checkpoint_step_N/training_state requires the
same task/GPU count. Publication completes after all model/optimizer shards and
rank RNG/loader states are written successfully.
