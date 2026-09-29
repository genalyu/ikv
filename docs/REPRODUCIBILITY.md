# Portable RGB-only IKV

### Serving policy is independent of the training recipe

Official weights trained with use_ikv_training=False can use IKV inference.
Post-training serve configs default to global KV policy in this repository;
set IKV_CACHE_POLICY=fifo explicitly for FIFO diagnostics. This does not change
the training recipe. Invalid policy values fail during configuration loading.

Check the actual [serve-contract] log, not just a launcher label or code version.
Dense global cold grounding must also log [cold-seed-rebuild]. A FIFO run is not
an IKV-global result, even if its files or task keys were named "ikv".


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

### Cold seed reconstruction

A clean-clamped seed has real input values, but its deeper cached representation
was computed alongside imagined future tokens. Dense RGB global-cache serving now
invalidates ALL cold cached entries on the first grounding and rebuilds the RGB
and action seed together with the real continuation, matching the released FIFO
grounding context. Later grounding still clears predictions only and preserves
real history. RGB-motion/sparse serving keeps its existing separate lifecycle.

The clear operation is transactional across every layer: if encoding or either
grounding expert fails, cached observations and predictions are restored.
The public clear_pred_cache API has a keyword-only include_observed=False option.
Deployment adapters must forward this option to every PP rank and preserve the
transaction snapshots. Do not deploy this server against an old PP clear hook.

The released cold dense layout prepends RGB/action but not tactile frames.
For that one mixed-length grounding, keep tactile descriptors but defer visual
contact association rather than invent same-time matches. No input latent,
image order, action values, temporal RoPE layout or success criterion is changed.

Fixed-request two-chunk diagnostics aligned rebuilt global-cache actions exactly
with FIFO and checked selected layer Q/K/V, cache, and attention output. This is
not a proof of task-score reproduction. Closed-loop results remain a separate gate.

The long-stream CPU regression compares 24 rounds of prediction, temporary
denoising and real grounding, at batch sizes 1 and 2, against FIFO without
eviction. It checks exact outputs and cache tensors/slot IDs. Full clearing uses
a copied selection mask so invalidating valid bits cannot skip ID/flag cleanup.
This regression does not establish full-checkpoint or task-score equivalence.
