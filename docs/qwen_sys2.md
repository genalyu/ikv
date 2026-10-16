# Qwen3.5 Sys2 scoring (opt-in inference)

This mode uses Qwen native vision merger features for both scene RGB and tactile
RGB. No DINO, NeoForce or force-cali representation is used for this index.
Sys1 observation encoding, model weights and action computation are unchanged.

## Protocol

- Visual index: time plus native Qwen feature packet reference.
- Tactile index: time plus native Qwen feature packet reference plus binary contact.
- Frame and patch-grid mappings are interface metadata. Qwen and WAN grids are
  different: native Qwen tokens are never averaged into WAN embeddings.
- Supply `obs["tactile_contact"]` as one explicit boolean for the whole batch or
  a flat binary array aligned with sensor/frame/row/column tactile latent patches.
  Scalar scope must genuinely cover all sensors and times in that batch. Missing,
  uncertain, nonbinary or misaligned labels raise an error. No force inversion or
  default-to-no-contact is performed. The caller owns reliable contact detection.
- Non-contact images retain their visual features. Release observations can
  themselves be important historical evidence.
- All archived real candidates and protected incoming real candidates enter each
  scoring request; no decoder KV or conversation history is reused across requests.
  Identical PP requests may share the last validated answer.
- English prompt is `n0_twam/preprocessing/sys2_prompt.txt`, based on the approved
  prompt with force fields replaced by tactile visual features and binary contact.
- Every candidate must occur exactly once in the correct JSON array with an integer
  score in [0,100]. A single complete Markdown JSON fence may be removed by the
  transport adapter; explanations, partial JSON and extra fields are rejected. Failure rolls back the grounding transaction. No formula fallback.

## Attention and storage

Current real observation rows go directly to the GPU. Remaining per-modality GPU
budgets select scored history. All real visual/tactile KV is retained in the CPU archive; only
missing selected tensors transfer back. Predictions are temporary and excluded
from the archive. Action KV follows baseline FIFO by insertion order in its GPU capacity
(`action_capacity`). It is neither scored nor archived/ recalled on CPU and has no
retention dependency on visual/tactile KV. Temporary action forwards borrow the
window; committed predicted action appends evict old action KV like baseline, and
clearing predictions does not resurrect evicted actions. Visual/tactile scores
are independently ranked. Existing formula mode remains the default.

Archived tensors use the CPU/archive overflow settings in `docs/cpu_kv_archive.md`.
The Qwen service stores immutable native vision packets on disk, with a separate
`--max-feature-bytes` hard limit. It does not delete evidence automatically. Use a
fresh feature directory for a new deployment/session if desired; do not remove a
store while clients retain references into it. Limit errors do not silently drop
history. Runtime index storage is separate from archived Sys1 K/V storage.

## Runtime

Use an isolated Python environment with torch and transformers==5.3.0. Keep the
Sys1 environment unchanged. Download official Qwen/Qwen3.5-4B assets and verify
file hashes with `python scripts/download_qwen_sys2.py --output /path/to/Qwen3.5-4B`.
The generated download manifest identifies the feature-cache model revision.
Start from repository root:

```sh
PYTHONPATH=. python scripts/serve_qwen_sys2.py \
  --model /path/to/Qwen3.5-4B --feature-directory /path/to/feature-store
```

The service defaults to loopback. Expose it to remote Sys1 ranks through SSH
forwarding. Both PP ranks must resolve the same service and feature store.

Generate a separate serving configuration (never overwrite the old one):

```sh
python scripts/create_cpu_archive_config.py \
  --source old-overrides.json --output sys2-overrides.json \
  --max-gib 8 --overflow disk --directory /path/to/kv-spill \
  --sys2-endpoint http://127.0.0.1:8097 --sys2-task 'Your task'
```

The serving reset uses the actual episode prompt. Dense observation indexing is
required for this mode. The caller must provide the contact field described above.
Images are processed individually with their temporal indices, not as a compressed
video stream. Each source image is encoded once and cached. Every scoring request
reconstructs a native multimodal sequence containing all referenced frame features.

The default image encoding uses 262144 pixels (~256 merged visual tokens) per
frame. Native position/grid metadata is preserved. The default scorer reserves
16384 output tokens within a 32768-token context. Complete requests that exceed
this configured bound fail explicitly; long-horizon coverage is not unlimited.
Scoring is synchronous, so decode latency directly affects control latency.

## Verification and limitations

`tests/test_sys2.py` covers binary contact, exact JSON coverage, modality separation,
WAN camera/tactile layout, complete-history scoring, old KV retention and rollback.
`validate_qwen_sys2.py` compares native pixel encoding logits against cached-feature
logits using real Qwen weights, then requests visual/tactile JSON scores. Its
synthetic image is a protocol smoke test, not a tactile accuracy measurement.
Task success, tactile discrimination and real-time latency require task rollouts.
