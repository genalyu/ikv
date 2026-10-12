# Optional CPU history KV archive

Inference-only mode for IKV version 2. Default remains `history_storage: evict`.
Use the new semantic checkout and runtime on each server; legacy checkouts do
not support these fields. Do not pass an archive override to training.

```json
{
  "history_storage": "cpu_archive",
  "archive_max_bytes": 8589934592,
  "archive_overflow": "disk",
  "archive_directory": "/absolute/local/path/kv-archive",
  "query_weight": 0.0,
  "action_query_weight": 0.0,
  "tactile_query_weight": 0.0
}
```

Place these inside `kv_retention` in the serving override. Existing budgets and
score parameters remain configurable. `scripts/create_cpu_archive_config.py`
creates a separate override from an exported config and refuses to replace it.

Current real observations always get GPU slots. Before each observed forward,
CPU history is scored using current policy metadata and content prototypes.
Select video and tactile first, then actions supported by the selected real
video/contact tactile or incoming observations. Selection reserves space for
all incoming modality tokens. IDs prevent duplication. Transfer only missing
selected KV, preserving already selected GPU slots. All layers use one plan.
Archived metadata stays on CPU; full archived KV is never copied to GPU for
ranking. A score change can recall previously displaced history.

Predicted and temporary denoising KV use a separate GPU tensor list, included
in current attention and cleared at grounding. They never consume historical
budgets or enter the archive. No DINO backfill is performed for this temporary
prediction list. Actual real grounding continues its usual metadata path.

Each PP rank stores only its owned layers. Archives, statistics, recalled GPU
slots and temporary predictions participate in the existing request rollback.
Reset/clear closes the archive and removes its unique episode spill directory.
No weights or active training process are changed.

The byte limit applies to resident archival **K/V tensors per rank**. Metadata,
GPU slot snapshots, current-chunk copy buffers, and records held by an active
rollback snapshot are additional memory. Disk overflow spills whole oldest
chunks into unique episode subdirectories, retaining metadata for scoring.
Disk retrieval loads a selected chunk on CPU and then gathers needed tokens.
It can exceed the limit temporarily by one chunk, and is slower than RAM.
`error` overflow raises MemoryError and rolls back the request without deleting
history. Disk errors also propagate and roll back; restart/kill can leave a
spill subdirectory, which must be cleaned after confirming the episode ended.

There is no fixed token-independent time bound for CPU scoring: all archived
metadata is evaluated each observed forward. Large episodes increase CPU cost.
The stored K/V are historical representations, not full-history recomputation.
Archive retrieval is currently synchronous; pinned CPU storage helps copies,
but overlapping GPU compute/transfer is not claimed.

Diagnostics: `transformer.get_kv_archive_stats(cache_name)` returns local
archived_tokens, resident_bytes, spilled_bytes, recalled_tokens, h2d_bytes and
d2h_bytes. Byte counters cover KV tensors only and exclude transaction copies.

Server paths:
- .177 code: /home/ubuntu/genalyu/n0twam-dual-deploy/ikv-score-push
- .177 Python: /home/ubuntu/genalyu/n0twam-dual-deploy/semantic-venv/bin/python
- .143 code: /home/user/n0twam-dual-deploy/ikv-semantic
- .143 Python: /home/user/n0twam-dual-deploy/semantic-venv/bin/python
- A100 code: /mnt/cfs/9wt59p/genalyu/ikv-semantic-worktree
- A100 Python: /mnt/cfs/9wt59p/genalyu/ikv-task-data/runtime/semantic-venv/bin/python

Validation uses actual tiny two-layer MoT forwards: current observation
reservation, history retention, prediction isolation, recall of exact archived
K/V, late request rollback, limit failure, disk spill/recall/cleanup, and
production two-rank PP parity. Full robot evaluation remains separate.
