# Stage 1 sparse-history heuristic

This is the sole formula backend. It replaces the former screenshot formula;
the old 0.9 threshold, fixed normalization and 10% random score mixture are removed.
Learned Qwen scoring remains a separate backend.

For historical patch j, compute three terms:

- Recency: exp(-max(t - t_j, 0) / 8).
- Contact: 1[modality is visual] * exp(-d_j / 8), with d_j = 0 when no contact is observed.
- Prototype-gated novelty: 1[rho_j >= 0.75] * exp(-max(tau_class - t_j, 0) / 8).

The score is the raw weighted sum of these terms. Inference always uses weights
(1, 1, 1), with no normalization and no random score mixing. All timestamps and
durations are world-time steps, not seconds.

DINO patch features are matched to the nearest content prototype by cosine
similarity. The prototype timestamp is its most recent true observation.
Missing or zero features and unmatched content have no prototype bonus.
Only true observations advance prototype history. Visual and tactile prototype
banks remain independent; tactile patch content comes from its native DINO feature
field. Tactile patches have no contact term. Action history retains its original
sliding-window/FIFO policy.

## Stage 1 training

Starting from pretrained System 1, sample independent nonnegative weights for
the three terms for each training masking decision. The default is Uniform[0, 2]
per term (mean 1); this distribution is an implementation choice because the
specification does not prescribe a distribution. Sample retained-slot quotas
through the existing visual/tactile quota sampler. This produces varied sparse,
noncontiguous histories while optimizing the existing flow-matching loss.

Mask-weight sampling must be reproducible under replay and activation checkpointing.
Inference/evaluation never samples heuristic weights. Stage 2 keeps the resulting
System 1 executor frozen and trains its separate learned scorer.

See [semantic patch indexing](semantic_patch_backends.md) and
[tactile sidecars](TACTILE_SEMANTIC_SIDECARS.md).
