# IKV source-of-truth and release rules

- Maintain the portable RGB-only core here. Use native RGB for visual and tactile
  images; do not add implicit red/blue swaps or require depth inputs.
- Motion detection remains disabled by default. RGB frame-difference is opt-in.
- A100 is the development checkout; GitHub main is the published source of truth.
  The two 4090 installations are deployments of a recorded core commit plus
  separate pipeline-parallel adapters. Never copy a deployment tree over main.
- Keep machine-specific PP launchers, SSH details, credentials, downloaded model
  weights, evaluation artifacts and absolute deployment paths out of commits.
- For each requested core change: inspect existing changes, implement narrowly,
  run relevant regressions, update documentation, commit and push to GitHub.
  Verify remote HEAD equals the intended commit. Report push failures explicitly;
  do not claim synchronization when only a local commit exists. Never force push.
- Deployment updates must record the core commit and tracked runtime-file hashes,
  preserve the PP adapter separately, and verify adapters against the core API.
- Do not claim official task-score reproduction from unit tests alone. Record
  model revision, per-task norms, RGB contract, inference knobs, seeds, simulation
  version, and any deployment adapter version for measured evaluations.
