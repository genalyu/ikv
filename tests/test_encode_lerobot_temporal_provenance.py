from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT_PATH = Path(__file__).parents[1] / "script" / "encode_lerobot_n0_latents.py"
SPEC = importlib.util.spec_from_file_location(
    "encode_lerobot_n0_latents_temporal_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
ENCODER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENCODER
SPEC.loader.exec_module(ENCODER)


def test_temporal_provenance_maps_causal_latents_to_chunk_end_frame_ids() -> None:
    provenance = ENCODER.build_wan_temporal_provenance(
        frame_ids=[0, 3, 6, 9, 12, 15, 18, 21, 24],
        latent_num_frames=3,
        temporal_stride=4,
    )

    assert provenance == {
        "schema_version": 1,
        "anchor_semantics": "causal_chunk_end",
        "temporal_stride": 4,
        "latent_anchor_indices": [0, 4, 8],
        "latent_anchor_frame_ids": [0, 12, 24],
    }


def test_temporal_provenance_refuses_an_unexplained_latent_layout() -> None:
    with pytest.raises(RuntimeError, match="Refusing to guess latent anchors"):
        ENCODER.build_wan_temporal_provenance(
            frame_ids=[0, 1, 2, 3, 4, 5],
            latent_num_frames=2,
            temporal_stride=4,
        )


def test_encode_video_returns_provenance_from_the_actual_latent_count() -> None:
    class FakeVAE:
        config = SimpleNamespace(
            latents_mean=[0.0],
            latents_std=[1.0],
            scale_factor_temporal=4,
        )

        def encode(self, video):
            assert video.shape[2] == 9
            posterior = SimpleNamespace(
                mean=torch.zeros(1, 1, 3, 1, 1, device=video.device)
            )
            return SimpleNamespace(latent_dist=posterior)

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(9)]
    result = ENCODER.encode_video(
        frames=frames,
        frame_ids=[0, 3, 6, 9, 12, 15, 18, 21, 24],
        vae=FakeVAE(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    flat_latent, latent_frames, latent_height, latent_width, provenance = result
    assert flat_latent.shape == (3, 1)
    assert (latent_frames, latent_height, latent_width) == (3, 1, 1)
    assert provenance["latent_anchor_indices"] == [0, 4, 8]
    assert provenance["latent_anchor_frame_ids"] == [0, 12, 24]
