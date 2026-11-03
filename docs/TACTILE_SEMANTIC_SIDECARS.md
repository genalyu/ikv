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
