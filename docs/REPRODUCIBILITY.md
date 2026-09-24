# Portable RGB-only IKV

## Source and deployment boundaries

A100 develops this repository; GitHub main publishes the same portable core.
Two-4090 PP deployments use that core plus external PP adapters. A model checkpoint
is not a code version: record both. Never publish a PP deployment directory as
this repository, and never assume an old file copy matches GitHub main.

For an experiment, check out an explicit commit (`git checkout <commit>`) and
record `git rev-parse HEAD` plus `git status --short`. Follow INSTALL.md for the
model environment and DEPLOY.md for checkpoint bundles/per-task normalization.
The repository does not include weights, DINO or NeoForce checkpoints.

## Supported input and defaults

- Visual and tactile images use native RGB, without implicit channel swaps.
- No depth maps, camera intrinsics or camera poses are required by this path.
- `use_rgb_motion_tokens=False`, `rgb_motion_online_preprocess=False`.
- `kv_cache_policy='global'`; optional frame difference uses RGB only.
- Checkpoint camera/tactile order, per-task norms and verbatim prompts must match.
- NeoForce/contact indexing is optional and needs its documented tactile force
  payload and local checkpoint; RGB-only does not mean tactile-free.
- Legacy RGB-D research modules are not part of the supported reproduction path.

Predictions are temporary across control chunks: real grounding clears predicted
KV before adding actual observations, preserving observed history. Clearing and
both grounding passes roll back together if any operation fails. See INDEXED_KV.md
for global retention and its experimental scoring settings.

## Verification

Run `python -m pytest -q tests` in the documented model environment, with pytest
installed. The sidecar unit tests stub optional LeRobot imports; they do not
validate training against an arbitrary installed LeRobot version. Actual training
requires the version specified in requirements.txt / POST_TRAINING.md.

Unit tests use tiny models/stubs and are not proof of official task success rates.
For closed-loop results record the source commit, checkpoint revision/hash,
per-task normalization hash, prompt, seed, simulation commit/assets, step limit,
action execution cadence, denoising steps, memory settings and RGB contract.

Before deployment, record hashes of tracked runtime files and the core commit;
keep the PP adapter hash separately. Stop only the relevant evaluation services
before updating them. Run adapter regression/smoke tests after a core API change.
Do not mix results from different source versions into one campaign.
