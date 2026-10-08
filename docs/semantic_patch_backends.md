# Semantic patch indexing

Code is implemented directly on .177:
`/home/ubuntu/genalyu/n0twam-dual-deploy/ikv-score-push`.

## Backends

| backend | class prototype features | prompt relevance |
|---|---|---|
| dinov2 | original DINOv2 patch features | disabled |
| dinov2_txt | DINOv2 ViT-L/14 with registers, raw patches | matching dino.txt adapted patches vs local half of text embedding |
| dinov3_txt | DINOv3 ViT-L/16 raw patches | matching dino.txt adapted patches vs local half of text embedding |
| siglip2 | fixed-resolution SigLIP2 post-normalized patch tokens | singleton patch passed through the learned attention pooling head vs text embedding |

One whole-image backbone pass returns every patch. No patch-by-patch image crops.
Prompts are encoded once and reused until the episode prompt changes. All models
are frozen; metadata changes retention decisions, not WAN content embeddings.
The existing online prototype clustering is shared by all backends. It assigns
content classes by cosine threshold; it is not instance tracking or named object
classification.

SigLIP2 singleton pooling is a dense similarity heuristic, **not a calibrated
per-patch probability**. NaFlex models are rejected because their variable patch
layout needs a separate spatial mapping implementation. Fixed-resolution SigLIP2
uses Transformers' `SiglipModel`; this is the model class name even for SigLIP2.
The directory/model identifier must contain `siglip2`.

Spatial pooling coefficients `a_ij` follow the existing adaptive-average/bilinear grid mapper.

DINO text weights are backbone-specific. The supported official released adapters
use ViT-L; do not attach these adapters to the original DINOv2-base checkpoint.

## Formula

Native visual patches receive `q_j`; a real WAN video token receives the spatially pooled `Q_i`:

```latex
q_j = \sigma\left(\frac{
\langle \widehat{z}^{\mathrm{text}}_j, \widehat{e}_{\mathrm{prompt}} \rangle-b}{\tau}
\right),\qquad
Q_i=\sum_j a_{ij}q_j,\qquad
S_i=w_T T_i+w_C C_i+w_R R_i+w_{\mathrm{task}}Q_i.
```

Default semantic task configuration: `task_weight=1.0`,
`task_temperature=0.07`, `task_bias=0.0`.
These are starting values, not benchmark-calibrated values.
Without semantic opt-in, `task_weight=0.0` preserves old checkpoint behavior.
Predicted video, action, and tactile tokens receive no task term.
Class threshold defaults to 0.9 and can be adjusted through
`features.content_threshold`; calibrate it independently for each backbone.

DINOv3 changes native patch size from 14 to 16. Features and relevance are
mapped to the unchanged WAN token grid through the same spatial pooling, then
flattened as frame/height/concatenated-camera-width. WAN tokens, latents and
policy output dimensions do not change.

## Task JSON configuration

Add one of these objects to `runtime.semantic_encoder` in a new task overlay.
Keep the task's existing `prompt`, cameras, dataset and checkpoint configuration.
Paths below are example asset destinations, not preinstalled weights.

DINOv2 + dino.txt:

```json
{
  "backend": "dinov2_txt",
  "repo": "/home/ubuntu/genalyu/models/dinov2",
  "backbone_weights": "/home/ubuntu/genalyu/models/dinov2_vitl14_reg4_pretrain.pth",
  "head_weights": "/home/ubuntu/genalyu/models/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
  "text_weights": "/home/ubuntu/genalyu/models/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
  "bpe": "/home/ubuntu/genalyu/models/bpe_simple_vocab_16e6.txt.gz",
  "image_size": [224, 224],
  "task_temperature": 0.07,
  "task_bias": 0.0
}
```

DINOv3 + dino.txt:

```json
{
  "backend": "dinov3_txt",
  "repo": "/home/ubuntu/genalyu/models/dinov3",
  "backbone_weights": "/home/ubuntu/genalyu/models/dinov3_vitl16_pretrain.pth",
  "head_weights": "/home/ubuntu/genalyu/models/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
  "bpe": "/home/ubuntu/genalyu/models/bpe_simple_vocab_16e6.txt.gz",
  "image_size": [224, 224],
  "task_temperature": 0.07,
  "task_bias": 0.0
}
```

The official DINO repositories must be local, clean, pinned Git checkouts with
their dependencies installed in the chosen runtime. v2 explicitly constructs
the official released adapter architecture and loads local weights, since its
hub text entrypoint otherwise always downloads them. v3 uses its official local
hub loader with all asset paths supplied.

SigLIP2:

```json
{
  "backend": "siglip2",
  "model": "/home/ubuntu/genalyu/models/siglip2-base-patch16-224",
  "image_size": [224, 224],
  "task_temperature": 0.07,
  "task_bias": 0.0
}
```

This must be a complete local HuggingFace checkpoint including tokenizer and
processor assets. The image size must be the checkpoint's native fixed size.
No loader silently downloads missing assets.

Add `"task_weight": 1.0` to the task's `features` object if desired.
Task training modes `ikv` and `motion_ikv` enable this score automatically when
a semantic backend is selected.

## Build and train

Use a **new task/work root** for each backend or scoring configuration; the
task fingerprint rejects old sidecars in a changed task root. After the normal
convert/RGB/tactile stages:

```bash
cd /home/ubuntu/genalyu/n0twam-dual-deploy/ikv-score-push
/home/ubuntu/genalyu/n0twam-dual-deploy/venv/bin/python -m n0_twam.task_pipeline.cli prepare --task /absolute/server/task.json --stage features --device cuda
/home/ubuntu/genalyu/n0twam-dual-deploy/venv/bin/python -m n0_twam.task_pipeline.cli prepare --task /absolute/server/task.json --stage pool
/home/ubuntu/genalyu/n0twam-dual-deploy/venv/bin/python -m n0_twam.task_pipeline.cli train --task /absolute/server/task.json --mode ikv --gpus 1
```

A complete data/training environment requires existing project dependencies
including lerobot and av. The current .177 inference environment lacks these
two optional data-processing packages.

Feature sidecars store dense `task_relevance[F,spatial]` and
`semantic_provenance`: backend, asset byte hashes, DINO repo revision,
prompt hash, grid sizes, normalization, resize and scoring parameters.
Training rejects mismatched caches and missing relevance. Checkpoint metadata
and exported serving overrides carry the same contract. Online serving uses
the configured backend and episode prompt; class features and task relevance
are produced together. External semantic indices must provide both arrays plus
`obs["kv_semantic_provenance"]` matching the configured encoder.

Selecting a new backend requires new feature sidecars and corresponding
training/checkpoint metadata; this does not modify the old 1500-step weights or
replace a currently running evaluation service.

## Validation limits

Offline tests cover both DINO adapter contracts, a real small Transformers
SigLIP architecture, camera/token ordering, prompt caching, feature sidecars,
sparse/dense training gathers, retention eviction, training gradients, and
future-label isolation. The full pretrained text-adapter assets have not been
installed or run on this server, so pretrained quality, peak VRAM and latency
remain unmeasured.

Official APIs:
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/dinotxt.py
- https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/dinotxt.py
- https://huggingface.co/google/siglip2-base-patch16-224
