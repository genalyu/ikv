# Online RGB frame difference

Implemented in `n0_twam/preprocessing/rgb_frame_difference.py` and routed by
`n0_twam/n0_twam_server.py`. Each camera is compared only with its own previous
RGB frame. Images are resized to model resolution; mean absolute RGB difference
is averaged over each WAN spatial patch. Threshold is in normalized [0,1] units.
Every changed patch is retained; the first observation is dense. Adjacent-frame
masks within a causal VAE block are unioned so transient movement is retained.
Camera grids are concatenated along width, matching the WAN address convention.

Enable before constructing the server:

```python
cfg.use_rgb_motion_tokens = True
cfg.rgb_motion_online_preprocess = True
cfg.rgb_motion_input_mode = 'rgb'
cfg.rgb_motion_rgb_threshold = 0.02
cfg.kv_cache_policy = 'fifo'  # existing sparse RGB path requires FIFO
```

The server derives capacity from the complete camera grid, overriding the old
32-token limit. Existing obs image requests are sufficient; no depth, intrinsics,
or poses are required. Episode reset clears the previous frames. Prepared state
uses the existing grounding transaction/rollback. Sparse padding is masked, not
input as valid content. Dense VAE encoding remains; sparsity starts at WAN patches.

DINO is retained solely to satisfy the existing semantic-index contract; it does
not select movement. The RGB-only detector needs no depth or geometry. Prediction
support uses the existing carry-forward mechanism; future frames are not observed.
Camera motion and lighting changes can select patches because there is no ego-motion
compensation. Existing checkpoint compatibility and GPU rollout success have not
been established. Existing source-lock manifests must be intentionally refreshed
before using a hash-locked benchmark launcher; do not bypass a mismatch.

Tests: test_rgb_frame_difference.py and test_rgb_frame_difference_server.py.
The service helper test extracts actual methods and substitutes a deterministic
DINO encoder; it is not a full checkpoint inference test.
