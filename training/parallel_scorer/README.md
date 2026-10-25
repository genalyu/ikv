# Parallel all-history IKV scorer

This is the **Qwen image/task encoders + new bidirectional interaction network**
route. It does not modify Qwen's causal or recurrent language layers to score
history. Qwen language processing is used only for the task description. There
is no generated JSON or autoregressive score decoding.

## Architecture and exact acceleration

1. Encode each newly observed RGB or RGB visuotactile image independently with
   a pinned Qwen3.5 visual encoder and merger. Store its features with IDs,
   timestamps, sensor/modality IDs and spatial coordinates.
2. Encode task text separately. Cache it while task and encoder weights stay fixed.
3. Project all history features and add their attached metadata.
4. Each interaction block performs **full bidirectional history self-attention**,
   task cross-attention and an FFN. Visual and tactile candidates share the
   self-attention; only selection quotas are separate.
5. A shared scalar head produces N scores for variable N. Map scores to native
   Sys1 KV addresses, then select visual/tactile quotas. Action KV keeps recency.

Flash SDPA avoids materializing N-by-N attention matrices. Query tiling always
uses **all** keys/values, and FFN tiling changes only scheduling. This preserves
the dense attention computation up to floating-point differences. It does not
make its quadratic arithmetic linear. The implementation rejects future inputs,
has no causal attention mask and no array-position embedding. Permuting candidates
together with their metadata permutes their scores; physical time/space remain
visible. The network can learn redundancy and cross-modal relationships, but
architecture alone does not guarantee it learns to keep the newest duplicate.

Only context-independent encoder outputs are reusable. Every decision recomputes
contextual history representations/scores because the task/current time/history
may change. Fixed inference encoder weights are sufficient; training may update
them. An immutable encoder revision prevents mixing latent features from different
weights. Changing the encoder requires rebuilding old feature caches.

No candidate pruning, summary-token bottleneck, local attention, quantization or
history eviction is used. History is retained in FeatureBank; save/load persists
it. The current implementation moves the complete scoring packet to GPU, so it
is not an out-of-core attention implementation. Insufficient memory raises an
error instead of silently truncating history. CPU/disk native KV paging remains
the existing Sys1 storage responsibility.

## Sizes and measured inference

| Preset | Width / blocks / heads | Interaction parameters | Total scorer parameters* |
|---|---|---:|---:|
| small | 512 / 4 / 8 | 16,816,128 | 18,281,985 |
| medium | 768 / 6 / 12 | 56,710,656 | 59,204,353 |

*Total includes projections, metadata and scoring head with image/text feature
widths both 1024. It excludes Qwen encoders. Actual encoder widths change the
projection count. The 17M/57M labels refer to interaction blocks, not the entire
vision-language system. Reusing only encoders does not retain the full original
0.8B/2B model as the history interaction network.

Measured 2026-10-25, idle RTX 4090 24GB, PyTorch 2.9.0+cu126, BF16,
batch 1, 32 task tokens, feature widths 1024, query tile 65536, FFN tile 2048,
2 warmups and 20 timed repeats. Random initialized weights/synthetic features.
Median wall-clock milliseconds, CUDA synchronized, model plus modality top-k:

| All-history candidates | small | medium |
|---:|---:|---:|
| 10,000 | 9.96 | 22.51 |
| 20,000 | 28.43 | 66.49 |
| 40,000 | 98.20 | 226.89 |
| 80,000 | 366.04 | 823.91 |

These are **not end-to-end robot latency or trained accuracy**. They exclude RGB
and task encoding, bank assembly, host/device transfer, native KV fetching and
Sys1 inference. Peak PyTorch allocated memory at 40k was approximately 440/682 MiB,
excluding encoder and Sys1 weights and CUDA allocator reservations.
Smaller query tiles reduce fallback memory but add launch overhead: 2048-query
tiles gave approximately 106/291 ms at 40k on this same GPU.

Further exact optimizations: keep feature tensors GPU-resident when possible,
encode only newly arrived frames, cache task features, benchmark tile sizes,
batch independent episodes only when throughput rather than latency matters.
Compilation/CUDA graphs are not enabled or claimed benchmarked here. Data-dependent
validation and dynamic N need explicit handling before graph capture.
More aggressive linear/sparse/summary architectures can retain stored history
but change all-pairs interaction; they require separate accuracy evaluation.

## Files

- model.py: exact dense interaction, chunking, shared head, modality quotas.
- encoders.py: Qwen visual features and task-only text; optional encoder training.
- prepare.py: frozen offline feature export from existing RGB teacher manifests.
- data.py: label decoding, cache contract, episode leakage checks, regression/ranking.
- train.py: cached-feature or raw-RGB SFT, validation and checkpoints.
- runtime.py: persistent feature bank and native KV selection adapter.
- infer.py: scoring CLI.
- evaluate.py: per-modality teacher agreement, MAE and linear-gain NDCG.
- rl.py / train_rl.py: without-replacement ordered selection policy and frozen
  N0 FM group-relative training; not text-token GRPO.
- benchmark.py: reproducible timed scorer benchmark; refuses occupied GPUs.
- rl.example.yaml: FM-training configuration template.

Use an environment with PyTorch SDPA; Qwen paths additionally require Transformers
with Qwen3.5 support, numpy, Pillow and the existing project dependencies.
Run commands from repository root. MODEL/DATA/RUN are placeholders.

## SFT

Existing teacher raw manifest records:
`id, episode_id, task, fps, query_frame, rgb_npy, target`.
RGB is uint8 [frames,3,256,256,3]; target is the existing serialized
default_score/blocks JSON, decoded to [observed_frames,3,8,8].
Split by episode **before** feature preparation.

```bash
python -m training.parallel_scorer.prepare --manifest DATA/train.jsonl \
  --model MODEL --revision BASE_HASH --output DATA/train-features
python -m training.parallel_scorer.prepare --manifest DATA/val.jsonl \
  --model MODEL --revision BASE_HASH --output DATA/val-features
python -m training.parallel_scorer.train \
  --manifest DATA/train-features/manifest.jsonl \
  --validation DATA/val-features/manifest.jsonl \
  --revision BASE_HASH --preset small --query-chunk 65536 --output RUN/small-new
```

Repeat with `--preset medium`. For end-to-end encoder tuning, pass original raw
manifests, `--raw-model MODEL --train-vision`, optionally `--train-text`.
Do not feed cached features to a trainable encoder. Encoder parameters have a
separate learning rate. SFT writes scorer, changed encoder components and optimizer
states at checkpoints and epoch boundaries; resume is supported. For an updated encoder,
export features using `--encoder-state RUN/encoders-0.pt` and revision
`BASE_HASH:update-N`, matching that epoch's scorer checkpoint. Pin the original base.

The historical teacher data is RGB-only. It supplies no tactile labels.
The feature-packet/model/runtime interfaces support tactile features, but actual
visuotactile supervision or aligned FM feedback must be supplied; no tactile
expertise is claimed from RGB teacher labels. The convenience raw-record reader
covers the existing three-camera RGB dataset. Custom tactile datasets must build
the explicit mixed-modality packet contract below.

## Packet and online integration

A torch-saved packet contains:
- features [N,Dv], text [L,Dt], times [N], xy [N,2];
- modalities [N] (visual=0, tactile=1), sensors [N];
- query_time (same unit as times), unique string ids [N], encoder_revision;
- optional targets [N] in [0,100], supervised bool [N].

Time is seconds for the supplied RGB reader; do not mix world-time steps and seconds.
Spatial coordinates are normalized [0,1]. Record sensor orientation/calibration
consistently in the dataset. This gives the network correspondences to learn;
it is not a hard-coded visual/tactile alignment algorithm.

```python
from training.parallel_scorer.runtime import FeatureBank, RuntimeScorer

bank = FeatureBank("BASE_HASH")
# Append only new observations; rows are never evicted.
bank.append(ids, features, times, xy, modalities, sensors, "BASE_HASH")
packet = bank.packet(task_features, query_time)
scorer = RuntimeScorer("RUN/small-new/scorer-0.pt")
result = scorer.score(
    packet, budgets=[visual_slots, tactile_slots],
    kv_candidate_ids=native_visual_tactile_kv_candidate_ids,
    kv_modalities=native_visual_tactile_modalities,
)
native_indices = result["selected_indices"]
# Existing Sys1 cache uses these native addresses; its action path is unchanged.
```

The supplied native ID list is in **actual Sys1 visual/tactile KV layout order**.
It may map several native slots to one scorer region. Selection budget applies
after mapping, in native slots. Never infer a one-to-one correspondence merely
because both encoders use patches. Mapping must account for temporal VAE encoding,
camera layout, patch sizes and native sensor-major tactile order. Preserve the
original native temporal/position IDs when gathering KV. Selection does not
authorize reindexing RoPE positions or moving an imagined token into real history.
Use eligible/protected masks for cache eligibility/mandatory entries; protected
slots count against quotas. Quotas cannot be negative or smaller than protected
sets. Below-budget histories return available candidates, without padding fake KV.

This module supplies the selector/adapter, not a replacement robot controller.
Integrate its selected native addresses at the existing retention boundary and
record that adapter version for closed-loop experiments.

## Post-training

```bash
python -m training.parallel_scorer.train_rl --config training/parallel_scorer/rl.example.yaml
```

Edit the template first. The runner refuses occupied GPUs before loading models. The RL manifest contains feature_path, id, episode_id,
n0_batch, query_frame, target_raw_frame, alignment_provenance and
video_candidate_map; tactile_candidate_map is required when tactile targets exist.
Maps cover every clean native token including the final target frame. Historical
entries are feature row indices; final target entries are -1. Tactile maps use
native sensor-major order, with a -1 final frame per sensor. No inferred mappings.

Four sampled without-replacement selections share the same FM noise/timestep
and budget per decision. Ordered Plackett-Luce log probabilities are exact for
the sampled paths; their union defines the selected set. Group advantages train
the scorer, with a centered-logit reference penalty (explicitly **not** subset KL).
Action tokens retain recency. Optional tactile scores are passed through a
backward-compatible extension of the existing final-phase scorer-support hook.
Earlier support phases remain unchanged. Constant group rewards give zero policy
advantage; monitor unique_sets and reward variance.

This initial RL runner updates the scorer using cached features. Trainable-encoder
RL, environment-success rewards and automatic joint retraining of Sys1 are not
implemented. SFT supports trainable encoders. Existing sparse Sys1 training stays
in its own pipeline. A full production experiment still needs aligned raw
visuotactile data, the existing Sys1 checkpoints, calibrated FM loss scales and
closed-loop evaluation; these are not fabricated by this implementation.

## Evaluation and tests

```bash
python -m training.parallel_scorer.evaluate --checkpoint RUN/small-new/scorer-0.pt \
  --manifest DATA/val-features/manifest.jsonl --output RUN/teacher-eval.json
python -m training.parallel_scorer.benchmark --preset small \
  --tokens 10000 20000 40000 80000 --query-chunk 65536 --repeats 20 \
  --output RUN/benchmark-small.json
pytest -q tests/test_parallel_scorer.py tests/test_scorer_grpo.py tests/test_ikv_training.py
# In the Qwen3.5 environment:
pytest -q training/parallel_scorer/test_qwen_encoders.py
```

Teacher agreement measures imitation, not useful memory or robot success.
For Neosim/Neoreal compare identical Sys1 weights, native-slot budgets, seeds and
observation/action settings across recency, heuristic, small and medium selectors.
Report success by task, failure modes, scorer p50/p95, feature encoding, KV gather
and complete action-loop latency separately. Record encoder/scorer/Sys1 revisions
and simulator/deployment adapter versions. Unit tests and random-weight speed
measurements do not establish task performance.


## Full-dataset experiment support

For frozen encoders, `episode_cache.py` writes each episode's observed RGB
features once, in exact BF16 bit storage, and constructs query views by slicing
only the causal prefix. Train and validation retain the original episode split;
test data is not required for preparation or tuning. This avoids repeatedly
encoding and storing overlapping prefixes. Interrupted preparation resumes at
episode boundaries with `--resume`; it never truncates long queries.

```bash
python -m training.parallel_scorer.episode_cache \
  --manifests DATA/train.ready.jsonl DATA/validation.ready.jsonl \
  --model MODEL --revision WEIGHT_HASH:linear-patch-v1 \
  --output DATA/episode-features --image-batch 96
python -m training.parallel_scorer.train \
  --manifest DATA/episode-features/train.ready.features.jsonl \
  --validation DATA/episode-features/validation.ready.features.jsonl \
  --revision WEIGHT_HASH:linear-patch-v1 --preset small \
  --output RUN/small --checkpoint-steps 25
# Same command with --resume restores model, optimizer, cursor and RNG state.
```

The encoder reuses the existing algebraically equivalent full-patch linear
projection optimization from the Qwen SFT implementation. Original weight names
and shapes remain unchanged. BF16 convolution/GEMM rounding can differ; record
the implementation in the encoder revision and rebuild features when switching.
Do not mix old feature files produced by a different pipeline. CPU FP32 tests
verify projection output and gradient equivalence; BF16 is not claimed bit-exact.

Training now saves atomic last-state checkpoints and handles SIGTERM at the next
optimizer boundary. Configuration and manifest hashes are checked on resume.
`last-state.pt` restores training; `last-scorer.pt`, `best-scorer.pt` and epoch
scorer checkpoints serve inference. Changed encoder states are saved alongside
their matching scorer. A trainable encoder revision includes the update number.

Metrics record data preparation time, optimizer-update elapsed time, token count,
gradient norm, cumulative training wall time, GPU allocation hours and peak
allocated GPU memory. Validation records elapsed time over the full validation
manifest. GPU hours include CPU/data waits within the training phase and are not
a measure of active CUDA-kernel time. Checkpoint/process overhead and full-stage
wall time should additionally be captured by the experiment launcher.

`profile.py` measures real-feature inference across history-length quantiles:
first packet I/O, H2D transfer, scorer, modality selection and selected-index
return. It reports p50/p95 over repeats. `profile_encoders.py` separately measures
new three-camera RGB encoding and cached/uncached task encoding. Neither measures
Sys1 KV fetch or closed-loop control. Report those separately before deployment.

```bash
python -m training.parallel_scorer.profile --checkpoint RUN/small/best-scorer.pt \
  --manifest DATA/episode-features/validation.ready.features.jsonl \
  --output RUN/small/latency.json
python -m training.parallel_scorer.profile_encoders --model MODEL \
  --revision WEIGHT_HASH:linear-patch-v1 --manifest DATA/validation.ready.jsonl \
  --output RUN/encoder-latency.json
```

Choose models using downstream quality, latency by history length, encoding
cost, training GPU hours, peak memory and cache size together. Teacher MAE alone
does not establish that a larger encoder or interaction network is preferable.


## Live W&B comparison

`wandb_bridge.py` uploads durable logs from a CPU-only authenticated host into a
single project. It creates four named model runs in one comparison group and a
separate shared-preencoding/hardware run. It backfills existing metrics, follows
new rows every five seconds, and resumes deterministic run IDs after restarting.
It does not modify or restart the GPU training processes.

```bash
python -m training.parallel_scorer.wandb_bridge --root RUN_ROOT \
  --entity ENTITY --project PROJECT --poll 5
```

Model runs expose encoder size, interaction preset, parameter count and common
training settings. Metrics include score loss, held-out teacher MAE, update/data
time, cumulative training GPU hours and peak VRAM. Inference profiling produces
p50/p95 curves against history token count and a latency table. Encoder costs
and final quality are uploaded when their reports become available. Shared
preencoding is accounted once in the infrastructure run.

The score-regression/ranking loss differs from the original full-Qwen JSON
cross-entropy loss; their absolute values are not directly comparable even
when both are kept in the same project. Compare equivalent downstream quality
and costs. `WANDB_STATUS.json` and `wandb-bridge/state.json` record upload state;
the CPU uploader needs monitoring independently of the GPU queue.
