# Qwen scorer: SFT and offline FM-reward GRPO

The source is managed here; existing training directories and jobs are not
modified by importing or running the new code. Run commands from the repository
root. Keep checkpoints, datasets, task configs, logs and credentials outside Git.

## Current contract

The checked-in SFT is a source snapshot of the existing Qwen3.5 scorer:
all original RGB frames 0..query, three cameras, 256x256 RGB, native 8x8 merged
patch grid; JSON `default_score/blocks` output. It has **no tactile-image input
or tactile/action score output**. Do not describe this checkpoint as a trained
vision-tactile scorer. Extending that contract requires new data and training.

SFT trains the vision encoder/merger plus language LoRA. GRPO v1 loads BOTH the
SFT adapter and its `vision_state.pt`, freezes vision, and trains language LoRA.
A second frozen SFT adapter is the KL reference; disabling adapters would give
the wrong reference. The frozen vision is shared between policy and reference.

## SFT

`train.py` and its import dependencies retain the running implementation:
full-history collation, assistant-only loss, context checks, chunked vision,
bounded delta backward, activation offload, grouped checkpointing, balanced
sampling, manual gradient synchronization, checkpoint/restart checks.

`sft_source_manifest.json` records the original source hashes. The portable
changes are an optional `final_dataset_marker` config path and an explicit
`filter_quality.py --dataset` argument instead of machine-specific path rewrites.
No active run configuration or output is copied.

Copy `sft.example.yaml` to your experiment directory and fill paths. Existing
data readiness and quality/hash markers are still required.

```bash
accelerate launch --num_processes 2 training/qwen_scorer/train.py --config /path/to/sft.yaml
```

`export_dataset.py` and `prepare_rgb.py` preserve the existing teacher-label
export/RGB preparation pipeline. The export is a provisional snapshot: it does
not independently certify all planned labels as complete. Existing context and
quality audits must produce the readiness markers before SFT.

`requirements.lock.txt` is the SFT environment snapshot, not an instruction to
change an existing environment.

## GRPO semantics

For one query:

1. Sample **4** complete JSON responses from the current Qwen policy, at
   temperature 1 with no top-k/top-p truncation.
2. Select one modality budget triple for the group; all candidates share it.
3. Call the original `Trainer.convert_input_format/_prepare_input_dict` ONCE,
   with a seeded noise draw and one fixed scheduler index across frames.
   Video/action schedulers retain their original shifts, so the same index does
   not imply identical physical sigma across modalities.
4. Map scores to native N0 clean-video tokens using explicit temporal/spatial
   provenance. For the final noisy video/action phases, select visual history
   with stable top-s. Action/tactile history uses fixed recency selection.
   Earlier phases keep their original retention plan; current Qwen scores are
   never applied to earlier decisions.
5. Run the frozen N0 MoT and call its original `Trainer.compute_loss` on the
   final frame only. This includes native FM scheduler weighting, action masks
   and tactile patch ordering. No complete denoising rollout is needed.
6. Reward = negative weighted, fixed-scale-normalized video/action/tactile loss.
   Group-normalize with a variance floor, then clipped completion-token GRPO
   plus SFT-reference KL. Average candidates within each frame and accumulate
   gradients across independent frames before one optimizer update. The final
   partial accumulation window uses its actual number of frames.

The original noise preparation may corrupt condition latents; this choice is
seeded and shared across candidates as well. CFG tactile dropping is disabled.
Invalid/truncated JSON receives a reward below the worst valid candidate by a
fixed margin. If every completion is invalid, relative reward supplies no
learning signal; logs expose this rather than pretending progress occurred.
Zero-variance groups have zero policy advantage but may still receive KL.

History budgets exclude simultaneous noisy target tokens. The v1 path performs
full-history reselection for the query, not irreversible online eviction.
GlobalTactile remains a prediction target when present. Initial production
support requires dense RGB, temporal patch size 1, aligned modalities,
LocalTactile cross-attention OFF and contact gate OFF. These latter branches
could expose clean target tactile information outside the history mask and
are rejected until a causal interface is implemented.

## Aligned dataset

Do not identify latent frames with original RGB frames by arithmetic guess.
Temporal compression/resampling and camera tiling must be verified.

Each manifest JSONL record contains the usual scorer fields:
`id, episode_id, task, query_frame, fps, rgb_npy`, plus:

- `n0_batch`: tensor-only .pt dictionary, already batched (batch size 1), using
  the existing N0 dataset tensor fields. It contains the observation prefix
  followed by exactly one expert target frame.
- `kv_coordinates`: JSON list [clean_video_token_count,4], in native N0
  clean-video token order. Each history row is [raw_frame,camera,row,column].
  Final target rows are all [-1,-1,-1,-1].
- `target_raw_frame`: strictly after `query_frame`.
- `alignment_provenance`: record of the source episode identity, source frame
  mapping, causal latent encoder provenance and camera/patch mapping.

`prepare_pairs.py` can export these bundles from the existing N0 dataset:

```bash
python -m training.qwen_scorer.grpo.prepare_pairs --n0-config task_config:cfg --pairs /path/to/pairs.jsonl --output /path/to/new-aligned-data
```

Each input pair contains `scorer_record`, `dataset_index`,
`target_latent_frame`, `latent_raw_frame_ids`,
`latent_source_max_raw_frame_ids`, `video_patch_coordinates` ([camera,row,col]
for each native spatial patch), and `alignment_provenance`.
Supply these from verified preprocessing metadata. The exporter refuses
future-observation leakage and unknown tensor layouts, preserves the full
prefix, and slices all supported modalities together. Multiple target indices
from one episode give independent query groups. Use episode-disjoint train and
validation manifests.

## Launch and validation

Create a NEW environment with `requirements-grpo.txt`; install
`lerobot==0.3.3 --no-deps` afterward as in the repository's existing training
setup. The combined production dependency stack and full checkpoint workload
must be checked when the sparse checkpoint is supplied; the CPU verification
below uses the existing separate N0 and Qwen environments.

Copy `grpo.example.yaml`, supply a completed SFT checkpoint, frozen sparse
checkpoint, the original task config module and aligned manifest. Replace
reward scales with fixed calibration statistics. Use idle GPUs only: the
launcher refuses devices with existing compute processes. It creates a fresh
output directory and does not resume/overwrite another run.

```bash
python -m training.qwen_scorer.grpo.train --config /path/to/grpo.yaml
```

The example stops after two optimizer updates. This is a single-process,
sequential-candidate implementation; device selection permits Qwen and N0 on
separate GPUs. It does not change any active SFT launchers or use distributed
process groups. Saved RL adapters, processor/config/vision and optimizer state live at the
checkpoint root, compatible with the existing `infer.py --adapter` interface.
Automatic RL resume and closed-loop success reward are not implemented.

CPU verification:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_scorer_grpo.py tests/test_masked_ikv_training.py tests/test_ikv_training.py -q
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest training/qwen_scorer/test_grpo_policy.py -q
```

The first suite uses the real tiny N0 MoT, original Trainer/scheduler/loss, and
tests shared noise, selection changes, causal masking, final-frame loss, frozen
weights and gradient accumulation. The second uses a real tiny Qwen3.5 hybrid
model/LoRA and checks output-token likelihoods, gradients and cached-generation
probability replay. These are CPU integration tests with random tiny weights,
not evidence of full-checkpoint training throughput or task success.
