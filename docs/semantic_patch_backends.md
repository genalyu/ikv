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
The example paths below are installed assets on .177. The same relative layout is installed on .143 and A100 (see deployment table).

DINOv2 + dino.txt:

```json
{
  "backend": "dinov2_txt",
  "repo": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/dinov2",
  "backbone_weights": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/dinov2_vitl14_reg4_pretrain.pth",
  "head_weights": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
  "text_weights": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
  "bpe": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/bpe_simple_vocab_16e6.txt.gz",
  "dtype": "bfloat16",
  "image_size": [224, 224],
  "task_temperature": 0.07,
  "task_bias": 0.0
}
```

DINOv3 + dino.txt:

```json
{
  "backend": "dinov3_txt",
  "repo": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/dinov3",
  "backbone_weights": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
  "head_weights": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
  "bpe": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/bpe_simple_vocab_16e6.txt.gz",
  "dtype": "bfloat16",
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
  "model": "/home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic/siglip2-base-patch16-224",
  "dtype": "bfloat16",
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
/home/ubuntu/genalyu/n0twam-dual-deploy/semantic-venv/bin/python -m n0_twam.task_pipeline.cli prepare --task /absolute/server/task.json --stage features --device cuda
/home/ubuntu/genalyu/n0twam-dual-deploy/semantic-venv/bin/python -m n0_twam.task_pipeline.cli prepare --task /absolute/server/task.json --stage pool
/home/ubuntu/genalyu/n0twam-dual-deploy/semantic-venv/bin/python -m n0_twam.task_pipeline.cli train --task /absolute/server/task.json --mode ikv --gpus 1
```

Separate semantic runtimes now include lerobot, av, timm and Transformers 4.57.6.
They inherit existing torch/torchvision and install their overrides in an
isolated environment. The active inference/training environments are unchanged.

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

## Validation

Offline tests cover both DINO adapter contracts, a real small Transformers
SigLIP architecture, camera/token ordering, prompt caching, feature sidecars,
sparse/dense training gathers, retention eviction, training gradients, and
future-label isolation. All three full pretrained backends passed validation on .177 with actual
Hidden USB top/wrist RGB: offline/online relevance agreement, prompt changes,
sidecar reload, class history, score-based eviction, and a tiny recurrent training
forward/backward. 220 relevant unit/integration tests passed. These checks do not
measure task success rate or constitute a full policy retraining/rollout.

Official APIs:
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/dinotxt.py
- https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/dinotxt.py
- https://huggingface.co/google/siglip2-base-patch16-224

## Installed server deployment

| server | code checkout | Python runtime | asset root |
|---|---|---|---|
| .177 | /home/ubuntu/genalyu/n0twam-dual-deploy/ikv-score-push | /home/ubuntu/genalyu/n0twam-dual-deploy/semantic-venv/bin/python | /home/ubuntu/genalyu/n0twam-dual-deploy/models/semantic |
| .143 | /home/user/n0twam-dual-deploy/ikv-semantic | /home/user/n0twam-dual-deploy/semantic-venv/bin/python | /home/user/n0twam-dual-deploy/models/semantic |
| A100 | /mnt/cfs/9wt59p/genalyu/ikv-semantic-worktree | /mnt/cfs/9wt59p/genalyu/ikv-task-data/runtime/semantic-venv/bin/python | /mnt/cfs/9wt59p/genalyu/ikv-task-data/shared-models/semantic |

New checkouts on .143/A100 preserve the older working checkouts and active runs.
Original DINOv2-base remains available on all three servers.
Assets use official DINOv2 downloads and public ModelScope copies for DINOv3/SigLIP2.
SHA256 hashes and source URLs are recorded in assets-manifest.json.
Official repository revisions: DINOv2 7764ea0; DINOv3 6876159.
DINOv3 loads local files directly without copying several GB into TORCH_HOME.
Provenance uses the Torch release version without the local CUDA build suffix;
the CUDA build is diagnostic information, rather than a different asset identity.

### Measurements on .177

RTX 4090; BF16; 224x224; batch of two cameras; five warm forwards.
This times the feature encoder, excluding policy inference and video decoding.

| backend | total parameters (vision + text) | peak allocated VRAM | median forward |
|---|---:|---:|---:|
| dinov2_txt | 867.8M | 1693.9 MiB | 5.75 ms |
| dinov3_txt | 866.6M | 1687.4 MiB | 7.52 ms |
| siglip2 | 375.2M | 751.8 MiB | 2.59 ms |

Report: /home/ubuntu/genalyu/n0twam-dual-deploy/semantic-validation/results/report.json.
Class threshold and task score temperature need task-specific quality evaluation.
These measurements do not establish which backend succeeds more often on robots.

### Ready task overlays on A100

/mnt/cfs/9wt59p/genalyu/ikv-task-data/semantic-tasks/hidden-usb/{dinov2_txt,dinov3_txt,siglip2}/task.json

Each uses a separate work root. Prepare all stages in the selected new task root,
then train using the unified task CLI. Do not reuse the old feature sidecars.

~~~bash
cd /mnt/cfs/9wt59p/genalyu/ikv-semantic-worktree
/mnt/cfs/9wt59p/genalyu/ikv-task-data/runtime/semantic-venv/bin/python -m n0_twam.task_pipeline.cli prepare --task /mnt/cfs/9wt59p/genalyu/ikv-task-data/semantic-tasks/hidden-usb/dinov3_txt/task.json --stage all --device cuda
/mnt/cfs/9wt59p/genalyu/ikv-task-data/runtime/semantic-venv/bin/python -m n0_twam.task_pipeline.cli check --task /mnt/cfs/9wt59p/genalyu/ikv-task-data/semantic-tasks/hidden-usb/dinov3_txt/task.json --mode ikv
/mnt/cfs/9wt59p/genalyu/ikv-task-data/runtime/semantic-venv/bin/python -m n0_twam.task_pipeline.cli train --task /mnt/cfs/9wt59p/genalyu/ikv-task-data/semantic-tasks/hidden-usb/dinov3_txt/task.json --mode ikv --gpus 4
~~~

scripts/create_semantic_tasks.py also creates these three overlays from another
supported source task. It preserves that task's prompt, robot, cameras and source.
Data adapter support remains the unified task pipeline's supported formats;
selecting a semantic backend alone does not add a new RoboDojo data adapter.

### Dual-host PP serving

scripts/semantic_pipeline.py supports IKV-v2 with replicated retention policies,
class history and task relevance. PP requires a newly trained compatible checkpoint
and the corresponding exported serve_overrides.json. Move model paths in the
export to each host's installed asset root; keep byte hashes/provenance intact.

Two-host CPU/Gloo validation on .177/.143 used actual two-layer MoT blocks.
The test compared outputs, each stage's KV, both retention policies and history
against a full-model baseline, with observed/predicted/action streams, eviction,
prediction clearing, transaction rollback/commit and episode reset.
Report: /home/ubuntu/genalyu/n0twam-dual-deploy/semantic-validation/pp-report.json.
This is a transport/retention check, not a full-size pretrained PP rollout.

On .177 use rank 0, GLOO_SOCKET_IFNAME=enp4s0, NCCL_SOCKET_IFNAME=enp4s0.
On .143 use rank 1 and both socket variables set to eno1.
Set IKV_SERVE_OVERRIDES to the per-host exported serving JSON,
TWAM_SERVE_OUT to a new log directory, and TWAM_PP_SPLIT (default 15).
IKV_DINO_DEVICE defaults to cpu; cuda:0 is available when there is VRAM headroom.

~~~bash
# Run from each host's code checkout and use that host's semantic Python.
python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=192.168.50.177 --master_port=30190 scripts/semantic_pipeline.py
# Same command on .143 with --node_rank=1.
~~~

Existing 1500-step policy weights remain the old training setup.
The new feature/scoring contract must be prepared and trained before claiming
evaluation results for the new method.
