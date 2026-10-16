

## Split-host streaming preparation

Run scripts/stream_robodojo_prepare.py with --role cpu --cpu-workers 35 on a100xl (120 CPU quota), and --role gpu on neosim-mem-a100 (four A100 GPUs), using the same shared --root. Conversion publishes per-episode completion markers only after atomic parquet and video outputs. Four persistent GPU workers claim episodes exclusively and cache VAE, text encoder and DINOv3 on device. Completed episodes are resumable. GPU coordination validates the complete corpus, waits for FOUR_GPU_PREFLIGHT_PASSED, then launches four-GPU IKV training. The external preflight watcher must run on the GPU host and perform two optimizer updates with a complete checkpoint before releasing training. No tactile or motion scoring is enabled.

RGB-only v2 retention accepts tactile_capacity=0; sampled capacity reserves slots only for enabled modalities and keeps tactile capacity at zero.
