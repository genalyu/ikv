from types import SimpleNamespace
import pytest
from torch.utils.data import DistributedSampler
from n0_twam.dataset.bucket_sampler import WithinUpdateDistributedSampler


class Pool:
    def __init__(self, n):
        self._datasets = [SimpleNamespace(
            used_video_keys=["head", "wrist"],
            new_metas=[dict(start_frame=0, end_frame=1 + (i * 7) % 29) for i in range(n)],
        )]

    def __len__(self):
        return len(self._datasets[0].new_metas)


@pytest.mark.parametrize("n,world,accum", [(1059, 8, 4), (5, 8, 4), (23, 2, 3)])
def test_keeps_reference_global_batches_across_epochs_and_resume(n, world, accum):
    data = Pool(n)
    def epochs(sampler_type):
        updates = []
        for epoch in range(6):
            ranks = []
            for rank in range(world):
                kw = dict(num_replicas=world, rank=rank, seed=42)
                if sampler_type is WithinUpdateDistributedSampler:
                    kw["accumulation_steps"] = accum
                s = sampler_type(data, **kw); s.set_epoch(epoch)
                ids = list(s); assert list(s)[2:] == ids[2:]
                ranks.append(ids)
            updates.extend(zip(*ranks))
        return updates
    original, grouped = epochs(DistributedSampler), epochs(WithinUpdateDistributedSampler)
    assert len(original) == len(grouped)
    for start in range(0, len(original), accum):
        assert sorted(sum(original[start:start+accum], ())) == sorted(sum(grouped[start:start+accum], ()))
    if n == 1059:
        def critical(rows):
            return sum(max(data._datasets[0].new_metas[i]["end_frame"] for i in row) for row in rows)
        assert critical(grouped) < critical(original)


def test_resume_rejects_different_layout_before_loading_shards(tmp_path):
    import json
    from n0_twam.task_pipeline.checkpoint import load_training_state
    cfg = SimpleNamespace(world_size=8, rank=0, task_mode="ikv", task_fingerprint="same",
                          ikv_sampler_layout="within_update_v1_accum4")
    # Legacy state has no sampler key and therefore used conventional ordering.
    (tmp_path/"complete.json").write_text(json.dumps(dict(
        world_size=8, mode="ikv", task_fingerprint="same",
    )))
    with pytest.raises(ValueError, match="ikv_sampler_layout mismatch"):
        load_training_state(SimpleNamespace(config=cfg), tmp_path)
