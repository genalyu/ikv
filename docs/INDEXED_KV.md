# Full-RGB indexed KV cache

This is the current RGB experiment. It preserves the N0-TWAM weights, content
embeddings, expert layers, output heads, and generation schedule. It adds one
independent index/statistics table and a global cache-retention policy. It does
not perform motion detection, object reconciliation, or prediction re-encoding
into the world model. Historical sparse experiments remain available with the
legacy `fifo` policy; they are not used by this mode.

## Enable and run

The shared defaults now enable this mode. In your chosen serving config, set a
local DINOv2 checkpoint (or an already cached Hugging Face model ID):

```python
cfg.use_rgb_motion_tokens = False
cfg.kv_cache_policy = 'global'
cfg.kv_index_dino_online = True
cfg.kv_index_predicted_dino = True
cfg.kv_index_dino_model_name_or_path = '/path/to/local/dinov2-base'
cfg.kv_index_dino_device = 'server'  # GPU; 'cpu' is also supported
cfg.kv_index_dino_image_size = (224, 224)
cfg.kv_retention = dict(
    top_k=128, time_scale=8.0,
    contact_weight=1.0, visual_weight=1.0, time_weight=1.0,
    query_weight=1.0, repetition_weight=1.0,
    action_scale=0.1, query_samples=16, seed=0,
)
```

Here `cfg` means your existing config object, such as `twam_server_cfg`.
Assemble the normal checkpoint bundle and use the same server/client launch
procedure in [DEPLOY.md](DEPLOY.md). No new transformer checkpoint or retraining
is required to run the mechanism. The retention weights are experimental, not
validated task-performance defaults.

DINO loading always uses `local_files_only=True`. No model is downloaded
implicitly. With predicted indexing enabled, the server needs DINO even if
observed DINO is supplied by the client. Decoding predicted RGB and running DINO
adds latency and memory usage; CPU VAE/DINO can be especially slow.

To test without DINO, disable **both** `kv_index_dino_online` and
`kv_index_predicted_dino`. Missing features remain unavailable and their visual
importance is zero. To restore the original cache policy entirely, select
`kv_cache_policy='fifo'`.

## Data flow and alignment

1. Real RGB is resized and encoded by the existing streaming WAN VAE. All WAN
   patches enter the original model; no motion mask or patch gather is used.
2. Frozen DINOv2 runs on real RGB at the causal endpoints represented by those
   latent frames. DINO features are pooled to each camera's WAN patch grid and
   flattened in exactly `(frame, row, concatenated-camera-column)` order.
3. Each committed video/action/tactile token receives independent metadata.
   World time comes from its existing RoPE grid, not its denoising timestep.
   The index never enters hidden states, Q, K, V, or a trainable projection.
4. Video generation keeps the original repeated forwards: intermediate current
   KV is temporary; the last video forward commits every layer's KV. The cold
   clamped RGB seed is labelled observed; future video positions are predicted.
5. After the video loop, decode the final latent **only for DINO annotation**.
   Cameras are decoded as separate batch elements, not as adjacent image regions.
   Pool output DINO to the same grid and label the already committed predicted
   Video KV. This does not allocate KV, run another transformer forward, or
   overwrite the real seed's DINO. Action KV is not assigned arbitrary image
   patches. The action loop then runs in its original order.
6. Real grounding uses the existing single video forward and single actual
   action/state forward, each committing KV with `update_cache=2`.

The current implementation requires WAN's 16x spatial compression, temporal
transformer patch size 1, and camera sizes divisible by the spatial token stride.
For 256x256 RGB and `(1,2,2)` transformer patches, the WAN grid is 8x8 per camera;
DINO-B/14 at 224x224 produces 16x16 features, pooled 2x2 per WAN token. Pooling
labels a corresponding image region, **not an exclusive causal region or object
identity**: both the VAE and attention mix information across positions.

Temporal correspondence is explicit:

- Real streaming input: one cold seed RGB; warm input contains complete VAE
  stride groups. With stride 4, warm latent endpoints are raw indices 3,7,... .
- Decoded cold prediction: latent endpoints correspond to RGB indices 0,4,... .
- Decoded continuation: prepend one most recent real latent as **decoder-only**
  context; use RGB indices 4,8,... and exclude the prefix from annotations.
  This bounded context is not identical to full-episode causal decoding.
- If `video_exec_step` truncates sampling, labels describe the resulting
  truncated output, not a guaranteed fully denoised sample. No extra world-model
  forward is added to refresh the saved KV after the last scheduler update.

## Index payload

The semantic fields are `{t, dino, neoforce, observation_flag}`. Keep DINO and
NeoForce separate; `observation_flag` is 1 for real and 0 for predicted. Internal
UIDs/grid positions are bookkeeping addresses, not xyz/object identity.

Optional precomputed observation payload:

```python
obs['kv_index'] = {
    'dino': dino_features,        # [N,Dv] or [1,N,Dv]
    'neoforce': neo_features,    # [N,Dt] or [1,N,Dt], optional
    'duration': interval_lengths,  # scalar or [N], optional, default 1
}
```

`N` covers the newly encoded real video tokens only, including all cameras, in
the order above. Do not resend the cold seed in the subsequent warm observation
payload. The default duration unit is one WAN world-time step, **not seconds**;
provide measured interval lengths if your scoring should use seconds. Feature
widths must stay consistent within a cache. Supplied DINO bypasses online DINO
for that real observation. No depth, camera poses, or intrinsics are required.

Exact all-zero vectors mean absent; presence is `(feature != 0).any(-1)`, not
`feature.sum() != 0`. No separate visual/tactile-valid fields are stored by the
global policy. An entirely absent modality initially has feature width zero;
its buffer becomes zero-filled when a known dimension first arrives. These are
metadata placeholders for existing tokens, never additional predicted-image KV.

By the experiment's input contract, nonzero stored NeoForce means contact.
There is **no automatically wired NeoForce encoder or contact calibration** in
this path. Provide already aligned/contact-gated NeoForce from your producer;
raw tactile latent values do not substitute for NeoForce. A standalone tactile
tail can be labelled with `obs['tactile_kv_index']` at real grounding, using the
same feature schema aligned to its actual tail-token count. No predicted
NeoForce is fabricated from generated tactile latents.

## One global retention policy

Each logical token occupies corresponding slots in every layer. Layers retain
different K/V tensors, while index/statistics are stored once. Capacity uses the
original allocation formula:

```text
capacity = floor(attn_window / 2)
           * (latent_token_per_chunk + action_token_per_chunk)
```

Those sizing terms include their configured tactile tails. They are **not**
expert quotas: all experts compete for the same capacity and global ranking.

The importance score is:

```text
contact_weight * contact
+ visual_weight * visual
+ time_weight * time
+ query_weight * query
- repetition_weight * repetition
```

| Term | Definition |
| --- | --- |
| Contact | Nonzero NeoForce interval duration represented by this token, normalized by the maximum among candidates. Never multiplied by layer count or diffusion iterations. No inferred object-track accumulator. |
| Visual | Maximum nonnegative cosine similarity to real DINO features at the latest observed video time `t0`; missing features score zero. |
| Time | `exp(-abs(token_time - t0) / time_scale)`, treating past and predicted future symmetrically. Only real video observations advance `t0`. |
| Query | Sampled attention mass divided by sampled-query exposure, then normalized across candidates. Up to `query_samples` conditional-batch queries per layer, averaged across heads/layers. Measured on committed forwards, not every temporary denoising call; this is a usage proxy, not a literal count of all queries. |
| Repetition | For action rows, `exp(-nearest_action_MSE / action_scale**2)` against previously cached and earlier current action vectors. Based on normalized input actions, not key magnitudes. Higher redundancy lowers priority. |

When space is needed, protect the global top-k **existing** tokens, then
uniformly randomly delete only the required number from the remainder. Incoming
tokens must fit: effective k is capped at `capacity - incoming_count`. An update
larger than capacity is rejected. All layers apply the same selection. Seed plus
committed revision makes the draw reproducible; temporary/failed forwards do not
advance that state.

**Real observations do not clear predicted KV.** Grounding appends new observed
KV alongside existing predictions, even at the same time/patch address. It does
not zero their K/V or index, invalidate their slots, or remove their DINO labels.
The old automatic prediction-clear call and transformer interfaces are removed.
Both sources compete in normal global capacity eviction. Explicit episode reset
and temporary/failed-forward rollback remain separate mechanisms. There is no
per-object replacement or “predicted versus observed” reconciliation rule.

## Safety and inspection

Output annotations address a cache-generation owner plus stable token UID, not
just reusable physical slots. Stale/evicted/reset handles are rejected. A full
grid check rejects mismatched frame/camera/patch ordering. One backfill updates
the shared metadata, without changing any layer's K/V or moving `t0`.

The server transaction includes video/action KV, retention statistics, and
annotation changes. DINO/decoder failures or a later action failure roll back
the request. Decoder-internal state is restored and the real streaming encoder
cache is not advanced by prediction annotation.

```python
snapshot = transformer.get_global_retention(cache_name)
print(snapshot['t0'], snapshot['token_uid'].shape)
print(snapshot['world_time_id'], snapshot['observation_flag'])
print(snapshot['components'], snapshot['score'])
```

The snapshot is detached; editing it does not edit the cache. `kind` is 0 video,
1 action, 2 tactile. The older `get_semantic_cache` API belongs to the legacy
sparse path; use `get_global_retention` for this mode.

## Verification

```bash
python -m pytest tests/test_global_kv_retention.py tests/test_dense_kv_index_server.py \
    tests/test_mot_sparse_padding.py tests/test_rgb_motion_server_helpers.py -q
```

Tests exercise tiny real two-layer MoT forwards and controlled VAE/DINO serving
stubs: unchanged outputs/KV without eviction, cross-expert global retention,
zero-sentinel contact intervals, two-sided time, temporary/failed rollback,
UID-safe backfill, camera/time alignment, and one annotation between the original
video and action loops. Full-checkpoint GPU latency, memory use, generated-label
quality, and closed-loop task success still need evaluation on the deployment
hardware.
