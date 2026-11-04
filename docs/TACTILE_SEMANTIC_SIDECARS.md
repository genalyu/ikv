# Native tactile semantic sidecars

Set `features.tactile_semantic_backend="dinov3"` with
`formula_variant="screenshot_v1"` and `random_weight=0.1`.

The builder reads native tactile RGB, runs the same frozen DINO encoder used for
visual indexing, and pools each sensor to a 4x4 grid matching 8x8 tactile VAE
latents with 2x2 transformer patches. Features are detached metadata, never added
to latent content or Q/K/V. The cache records `dinov3-native-rgb`; it does not
claim to be a NeoForce model output.

The sidecar stores tactile features as [world frame, sensor, 16 patches, D].
Loader validation checks sensor order and frame alignment. Training reorders to
sensor-major, frame-major, patch-major to match global tactile tokens. The
historically named retention `neoforce` field transports these tactile semantic
vectors; visual DINO vectors remain in their separate field.

Measured contact comes from the raw per-sensor `contact_force` stream, aligned
causally to converted 30Hz frames. Consecutive nonzero-force frames divided by
12 give duration in WAN world-time steps. Each tactile patch uses its sensor's
contact duration; visual patches use the duration of any-sensor contact. This is
a frame/sensor contact annotation, not a visual spatial contact localization.
No oracle target IDs or actor poses enter semantic features.

Real tactile observations update an independent tactile content history.
Predicted branches carry zero semantic/contact metadata. Missing real tactile
sidecars fail task preflight when this feature backend is requested.

Raw tactile filenames are resolved from the converted episode's audited
source_provenance.jsonl mapping. Finalized conversion markers may lack legacy
source_id fields; a conflicting marker or missing/duplicate provenance fails
explicitly. New conversion markers retain source_id.


When force is absent, contact is explicitly estimated from native tactile depth: per-pixel temporal 95th-percentile reference, indentation above 0.1 mm on at least 0.5% of pixels. Sidecars record contact_provenance per sensor. This relative estimate may miss sustained contact and is not measured force. Measured-force inputs keep their existing behavior. Feature extraction supports disjoint IKV_FEATURE_SHARD_COUNT/ID trajectory modulo partitions; finalize once after all shards succeed.

Tactile VAE uses the same exact source stride-select grid as RGB (not FFmpeg fps resampling). Both retain F modulo 4 equal to 1; preflight compares tactile/RGB frame IDs and latent counts. Legacy fps-resampled tactile latents must be rebuilt, including apparently equal-length files, because source pixels may differ.

Dataset construction retries transient filesystem EIO at most five times with staggered backoff. Persistent I/O and invalid datasets still fail the required full-corpus guard; this never authorizes dropping tasks.

Immutable local episode Parquet is read directly into Arrow with the same LeRobot torch transform, bypassing shared Hugging Face builder/cache locks. This prevents lock reentrancy after a shared-filesystem read error and does not download or omit episode rows.
