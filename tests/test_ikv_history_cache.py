import torch

from n0_twam.models.global_kv_retention import RetentionConfig
from n0_twam.models.multimodal_kv_retention import ContentHistory


def test_cached_content_decisions_preserve_greedy_state():
    config = RetentionConfig(version=2, video_capacity=1, action_capacity=1,
                             tactile_capacity=1, content_capacity=2,
                             content_threshold=0.9)
    reference = ContentHistory(config, "cpu")
    recording = ContentHistory(config, "cpu")
    replay = ContentHistory(config, "cpu")
    cache = {}
    recording.decision_cache = cache
    replay.decision_cache = cache
    frames = [
        torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]),
        torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
        torch.tensor([[0., 0., 1., 0.], [0., 1., 0., 0.]]),
    ]
    for frame_id, values in enumerate(frames):
        times = torch.full((len(values),), float(frame_id))
        durations = torch.arange(1, len(values) + 1).float()
        for history in (reference, recording, replay):
            history.observe(values, times, durations)
        for field in ("features", "duration", "last_time"):
            expected = getattr(reference, field)
            assert torch.equal(getattr(recording, field), expected)
            assert torch.equal(getattr(replay, field), expected)
    assert replay.cache_hits == len(frames)
    assert len(cache) == len(frames)
