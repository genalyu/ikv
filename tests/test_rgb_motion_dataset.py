"""Unit tests for RGB-motion sidecar normalization.

The project keeps LeRobot as a post-training dependency rather than a core
package dependency.  These tests load the dataset module with tiny import
stubs when LeRobot is absent so the pure sidecar helpers remain testable in a
minimal development environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _load_dataset_module():
    module_path = (
        Path(__file__).parents[1] / "n0_twam" / "dataset" / "lerobot_latent_dataset.py"
    )

    inserted = []
    try:
        import lerobot  # noqa: F401
    except ModuleNotFoundError:
        stubs = {
            "lerobot": types.ModuleType("lerobot"),
            "lerobot.datasets": types.ModuleType("lerobot.datasets"),
            "lerobot.datasets.lerobot_dataset": types.ModuleType(
                "lerobot.datasets.lerobot_dataset"
            ),
            "lerobot.datasets.utils": types.ModuleType("lerobot.datasets.utils"),
            "lerobot.datasets.compute_stats": types.ModuleType(
                "lerobot.datasets.compute_stats"
            ),
            "lerobot.constants": types.ModuleType("lerobot.constants"),
            "lerobot.datasets.video_utils": types.ModuleType(
                "lerobot.datasets.video_utils"
            ),
        }
        stubs["lerobot.datasets.lerobot_dataset"].LeRobotDataset = object
        stubs["lerobot.datasets.lerobot_dataset"].LeRobotDatasetMetadata = object
        stubs["lerobot.datasets.utils"].get_episode_data_index = lambda *a, **k: None
        stubs["lerobot.datasets.compute_stats"].aggregate_stats = lambda *a, **k: None
        stubs["lerobot.constants"].HF_LEROBOT_HOME = Path("/tmp")
        stubs["lerobot.datasets.video_utils"].decode_video_frames = lambda *a, **k: None
        for name, stub in stubs.items():
            sys.modules[name] = stub
            inserted.append(name)

    try:
        spec = importlib.util.spec_from_file_location(
            "_rgb_motion_dataset_under_test", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in reversed(inserted):
            sys.modules.pop(name, None)


DATASET_MODULE = _load_dataset_module()
Dataset = DATASET_MODULE.LatentLeRobotDataset


def test_frame_local_sidecar_is_truncated_and_defaults_are_explicit():
    payload = {
        "motion_indices": torch.tensor(
            [[0, -1, -1], [3, 5, -1], [7, 8, 9], [2, -1, -1]]
        ),
        "dino_features": torch.arange(4 * 3 * 2).reshape(4, 3, 2).float(),
    }

    # Build the full latent timeline before cropping. Raw source-frame ids are
    # intentionally unrelated to the semantic WAN-step ordinal.
    full_world_times = Dataset._latent_world_time_ids(
        [100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136],
        4,
    )
    assert full_world_times.tolist() == [0, 1, 2, 3]
    out = Dataset._normalize_rgb_motion_payload(
        payload,
        expected_full_frames=4,
        spatial_tokens_per_frame=12,
        latent_world_time_ids=full_world_times,
        truncate_start=1,
        truncate_end=3,
    )

    assert set(out) == {
        "motion_indices",
        "motion_valid_mask",
        "motion_scores",
        "world_time_id",
        "dino_features",
        "neoforce_features",
        "observation_flag",
        "visual_valid",
        "tactile_valid",
    }
    assert out["motion_indices"].tolist() == [[3, 5, -1], [7, 8, 9]]
    # Crop [1:3] preserves the full segment's WAN ordinals; it is not reset to 0.
    assert out["world_time_id"].tolist() == [[1, 1, -1], [2, 2, 2]]
    assert out["observation_flag"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert out["neoforce_features"].shape == (2, 3, 0)
    assert not out["tactile_valid"].any()
    assert torch.equal(out["visual_valid"], out["motion_valid_mask"])
    # Padding features are deterministic zeros rather than stale sidecar data.
    assert torch.equal(out["dino_features"][0, 2], torch.zeros(2))


def test_explicit_global_indices_are_grouped_by_frame_without_guessing():
    payload = {
        # S=8 -> frames/spatial positions: (0,0), (1,1), (1,3), (2,7)
        "rgb_motion_indices": torch.tensor([0, 9, 11, 23]),
        "dino_features": torch.arange(8).reshape(4, 2).float(),
        "motion_scores": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "observation_flag": torch.tensor([1, 0, 1, 0]),
    }

    out = Dataset._normalize_rgb_motion_payload(
        payload,
        expected_full_frames=3,
        spatial_tokens_per_frame=8,
        latent_world_time_ids=torch.tensor([0, 1, 2]),
    )

    assert out["motion_indices"].tolist() == [[0, -1], [1, 3], [7, -1]]
    assert out["motion_valid_mask"].tolist() == [
        [True, False],
        [True, True],
        [True, False],
    ]
    assert out["observation_flag"].tolist() == [[1, 0], [0, 1], [0, 0]]
    assert torch.allclose(
        out["motion_scores"],
        torch.tensor([[0.1, 0.0], [0.2, 0.3], [0.4, 0.0]]),
    )
    assert out["dino_features"][1].tolist() == [[2.0, 3.0], [4.0, 5.0]]


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"motion_indices": torch.zeros(2, 1, dtype=torch.long)}, "dino_features"),
        (
            {
                "motion_indices": torch.tensor([[0], [8]]),
                "dino_features": torch.zeros(2, 1, 4),
            },
            "outside",
        ),
        (
            {
                "motion_indices": torch.tensor([[0]]),
                "dino_features": torch.zeros(1, 1, 4),
            },
            "frame mismatch",
        ),
    ],
)
def test_bad_sidecars_fail_with_actionable_errors(payload, error):
    with pytest.raises((KeyError, ValueError), match=error):
        Dataset._normalize_rgb_motion_payload(
            payload,
            expected_full_frames=2,
            spatial_tokens_per_frame=8,
            latent_world_time_ids=torch.tensor([0, 1]),
        )


def test_default_world_time_is_wan_latent_ordinal_not_raw_source_frame():
    assert Dataset._latent_world_time_ids(
        [20, 21, 22, 23, 24, 25, 26, 27, 28], 3
    ).tolist() == [0, 1, 2]
    # Even a metadata format with exactly one raw id per latent must not leak
    # those raw ids into the semantic world-time namespace.
    assert Dataset._latent_world_time_ids(
        [200, 900, 1500], 3
    ).tolist() == [0, 1, 2]


def test_explicit_sidecar_world_time_overrides_default_wan_ordinals():
    out = Dataset._normalize_rgb_motion_payload(
        {
            "motion_indices": torch.tensor([[1], [2], [3]]),
            "dino_features": torch.ones(3, 1, 4),
            "world_time_id": torch.tensor([41, 43, 47]),
        },
        expected_full_frames=3,
        spatial_tokens_per_frame=8,
        latent_world_time_ids=Dataset._latent_world_time_ids(
            [100, 104, 108], 3
        ),
    )
    assert out["world_time_id"].squeeze(1).tolist() == [41, 43, 47]


def test_fixed_k_sorts_by_score_then_pads_every_field():
    payload = {
        "motion_indices": torch.tensor([[1, 2, 3], [4, -1, -1]]),
        "motion_scores": torch.tensor([[0.2, 0.9, 0.4], [0.5, 8.0, 9.0]]),
        "dino_features": torch.tensor(
            [[[10.0], [20.0], [30.0]], [[40.0], [50.0], [60.0]]]
        ),
    }

    out = Dataset._normalize_rgb_motion_payload(
        payload,
        expected_full_frames=2,
        spatial_tokens_per_frame=8,
        latent_world_time_ids=torch.tensor([0, 1]),
        max_tokens_per_frame=2,
    )

    assert out["motion_indices"].tolist() == [[2, 3], [4, -1]]
    assert torch.allclose(out["motion_scores"], torch.tensor([[0.9, 0.4], [0.5, 0.0]]))
    assert out["dino_features"].squeeze(-1).tolist() == [
        [20.0, 30.0],
        [40.0, 0.0],
    ]


@pytest.mark.parametrize("index_key", ["motion_indices", "rgb_motion_indices"])
def test_empty_motion_is_represented_by_fixed_padding(index_key):
    if index_key == "motion_indices":
        indices = torch.empty((2, 0), dtype=torch.long)
        dino = torch.empty((2, 0, 4))
    else:
        indices = torch.empty((0,), dtype=torch.long)
        dino = torch.empty((0, 4))

    out = Dataset._normalize_rgb_motion_payload(
        {index_key: indices, "dino_features": dino},
        expected_full_frames=2,
        spatial_tokens_per_frame=8,
        latent_world_time_ids=torch.tensor([0, 1]),
        max_tokens_per_frame=3,
    )

    assert out["motion_indices"].shape == (2, 3)
    assert (out["motion_indices"] == -1).all()
    assert not out["motion_valid_mask"].any()
    assert out["dino_features"].shape == (2, 3, 4)


def test_zero_width_neoforce_never_claims_tactile_validity():
    out = Dataset._normalize_rgb_motion_payload(
        {
            "motion_indices": torch.tensor([[1]]),
            "dino_features": torch.ones(1, 1, 4),
            "neoforce_features": torch.empty(1, 1, 0),
            "tactile_valid": torch.zeros(1, 1),
        },
        expected_full_frames=1,
        spatial_tokens_per_frame=8,
        latent_world_time_ids=torch.tensor([0]),
    )
    assert out["neoforce_features"].shape == (1, 1, 0)
    assert not out["tactile_valid"].any()


def test_zero_width_neoforce_rejects_true_tactile_presence():
    with pytest.raises(ValueError, match="cannot be true"):
        Dataset._normalize_rgb_motion_payload(
            {
                "motion_indices": torch.tensor([[1]]),
                "dino_features": torch.ones(1, 1, 4),
                "neoforce_features": torch.empty(1, 1, 0),
                "tactile_valid": torch.ones(1, 1),
            },
            expected_full_frames=1,
            spatial_tokens_per_frame=8,
            latent_world_time_ids=torch.tensor([0]),
        )


def test_nonempty_neoforce_requires_explicit_presence_mask():
    with pytest.raises(KeyError, match="missing tactile_valid"):
        Dataset._normalize_rgb_motion_payload(
            {
                "motion_indices": torch.tensor([[1]]),
                "dino_features": torch.ones(1, 1, 4),
                "neoforce_features": torch.zeros(1, 1, 3),
            },
            expected_full_frames=1,
            spatial_tokens_per_frame=8,
            latent_world_time_ids=torch.tensor([0]),
        )


def test_duplicate_valid_motion_addresses_are_rejected_per_frame():
    with pytest.raises(ValueError, match="duplicate valid addresses"):
        Dataset._normalize_rgb_motion_payload(
            {
                "motion_indices": torch.tensor([[2, 2]]),
                "dino_features": torch.ones(1, 2, 4),
            },
            expected_full_frames=1,
            spatial_tokens_per_frame=8,
            latent_world_time_ids=torch.tensor([0]),
        )


@pytest.mark.parametrize(
    "feature,valid_field",
    [
        ("dino_features", "visual_valid"),
        ("neoforce_features", "tactile_valid"),
    ],
)
def test_present_semantic_features_must_be_finite(feature, valid_field):
    payload = {
        "motion_indices": torch.tensor([[1]]),
        "dino_features": torch.ones(1, 1, 4),
        "neoforce_features": torch.ones(1, 1, 3),
        "visual_valid": torch.ones(1, 1),
        "tactile_valid": torch.ones(1, 1),
    }
    payload[feature][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match=f"{feature}.*finite"):
        Dataset._normalize_rgb_motion_payload(
            payload,
            expected_full_frames=1,
            spatial_tokens_per_frame=8,
            latent_world_time_ids=torch.tensor([0]),
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("motion_indices", torch.tensor([[1.9], [2.0]])),
        ("world_time_id", torch.tensor([[0.0], [4.5]])),
        ("observation_flag", torch.tensor([[1.0], [0.9]])),
    ],
)
def test_fractional_index_fields_are_rejected(field, value):
    payload = {
        "motion_indices": torch.tensor([[1], [2]]),
        "dino_features": torch.ones(2, 1, 4),
        field: value,
    }
    with pytest.raises(ValueError, match="integer-valued|only 0 or 1"):
        Dataset._normalize_rgb_motion_payload(
            payload,
            expected_full_frames=2,
            spatial_tokens_per_frame=8,
            latent_world_time_ids=torch.tensor([0, 1]),
        )


def test_shared_hook_is_a_strict_noop_when_feature_is_disabled():
    dataset = object.__new__(Dataset)
    dataset.use_rgb_motion_tokens = False
    sample = {"latents": torch.zeros(2, 16, 48, 4)}
    before_keys = set(sample)

    dataset._maybe_load_rgb_motion(
        sample,
        episode_index=0,
        local_start_frame=0,
        local_end_frame=9,
        expected_full_frames=2,
        latent_world_time_ids=None,
    )

    assert set(sample) == before_keys


def test_shared_hook_uses_concatenated_camera_patch_grid():
    dataset = object.__new__(Dataset)
    dataset.use_rgb_motion_tokens = True
    dataset.used_video_keys = ["left", "right"]
    dataset.config = SimpleNamespace(patch_size=(1, 2, 2))
    captured = {}

    def fake_load(_self, **kwargs):
        captured.update(kwargs)
        return {"motion_indices": torch.tensor([[0], [1]])}

    dataset._load_rgb_motion_sidecar = types.MethodType(fake_load, dataset)
    sample = {"latents": torch.zeros(2, 16, 48, 4)}
    world_times = torch.tensor([0, 1])
    dataset._maybe_load_rgb_motion(
        sample,
        episode_index=3,
        local_start_frame=0,
        local_end_frame=9,
        expected_full_frames=2,
        latent_world_time_ids=world_times,
        latent_frame_ids=torch.tensor([0, 1, 2, 3, 4]),
        camera_wan_grid_shapes={"left": (8, 8), "right": (8, 16)},
    )

    # H'=16/2=8, W'_total=48/2=24: one frame has 8*24 patch addresses.
    assert captured["spatial_tokens_per_frame"] == 192
    assert captured["camera_wan_grid_shapes"] == {
        "left": (8, 8),
        "right": (8, 16),
    }
    assert captured["latent_world_time_ids"] is world_times
    assert "motion_indices" in sample


def test_meta_validation_checks_sidecar_presence_readability_and_frame_count(
    tmp_path,
):
    dataset = object.__new__(Dataset)
    dataset.root = tmp_path
    dataset.config = SimpleNamespace(
        use_rgb_motion_tokens=True,
        rgb_motion_root_name="rgb_motion",
        rgb_motion_max_tokens=32,
        tactile_latent_root_name="latents_tactile",
    )
    dataset.meta = SimpleNamespace(
        info={"chunks_size": 1000},
        get_episode_chunk=lambda _episode: 0,
    )
    dataset.used_video_keys = ["cam"]
    dataset.used_tactile_keys = []
    dataset.has_tactile_condition = False
    dataset.synthetic_tactile_data = False
    dataset.filter_mismatched_latents = True
    dataset.use_rgb_motion_tokens = True
    dataset._latent_frame_count_cache = {}
    dataset._rgb_motion_meta_cache = {}
    dataset._meta_filter_counts = {}

    latent_file = tmp_path / "video.pth"
    torch.save({"latent_num_frames": 2}, latent_file)
    dataset._resolve_latent_file = types.MethodType(
        lambda _self, *_args: latent_file, dataset
    )

    # Missing while enabled is filtered with an explicit reason.
    assert not dataset._check_meta(0, 9, 0)
    assert dataset._meta_filter_counts == {"missing_rgb_motion_sidecar": 1}

    sidecar_file = tmp_path / "rgb_motion" / "chunk-000" / "episode_000000_0_9.pth"
    sidecar_file.parent.mkdir(parents=True)
    torch.save(
        {
            "motion_indices": torch.tensor([[0]]),
            "dino_features": torch.ones(1, 1, 4),
        },
        sidecar_file,
    )
    dataset._rgb_motion_meta_cache.clear()
    assert not dataset._check_meta(0, 9, 0)
    assert dataset._meta_filter_counts["rgb_motion_video_frame_mismatch"] == 1

    torch.save(
        {
            "motion_indices": torch.tensor([[0], [1]]),
            "dino_features": torch.ones(2, 1, 4),
        },
        sidecar_file,
    )
    dataset._rgb_motion_meta_cache.clear()
    assert dataset._check_meta(0, 9, 0)

    signature = dataset._repo_validation_signature()
    assert signature["rgb_motion_enabled"] is True
    assert signature["rgb_motion_root_name"] == "rgb_motion"
    assert signature["rgb_motion_sidecars"] == 1
    assert signature["rgb_motion_max_tokens"] == 32


def test_multicamera_sidecar_requires_address_provenance(tmp_path):
    dataset = object.__new__(Dataset)
    dataset.used_video_keys = ["left", "right"]
    dataset.config = SimpleNamespace(patch_size=(1, 2, 2))
    dataset._rgb_motion_meta_cache = {}
    sidecar = tmp_path / "sidecar.pth"
    base = {
        "motion_indices": torch.tensor([[0]]),
        "dino_features": torch.ones(1, 1, 4),
    }

    torch.save(base, sidecar)
    with pytest.raises(KeyError, match="address provenance"):
        dataset._rgb_motion_sidecar_summary(sidecar)

    dataset._rgb_motion_meta_cache.clear()
    torch.save(
        {
            **base,
            "camera_keys": ["right", "left"],
            "patch_size": (1, 2, 2),
            "spatial_grid_shape": (8, 16),
            "provenance": {
                "camera_wan_grid_shapes": {
                    "right": [8, 8],
                    "left": [8, 8],
                }
            },
        },
        sidecar,
    )
    with pytest.raises(ValueError, match="camera order mismatch"):
        dataset._rgb_motion_sidecar_summary(sidecar)

    dataset._rgb_motion_meta_cache.clear()
    torch.save(
        {
            **base,
            "camera_keys": ["left", "right"],
            "patch_size": (1, 2, 2),
            "spatial_grid_shape": (8, 16),
            "provenance": {
                "camera_wan_grid_shapes": {
                    "left": [8, 8],
                    "right": [8, 8],
                }
            },
        },
        sidecar,
    )
    assert dataset._rgb_motion_sidecar_summary(sidecar)["frame_count"] == 1


def test_multicamera_latents_must_share_exact_source_frame_ids():
    dataset = object.__new__(Dataset)
    dataset.used_video_keys = ["left", "right"]
    dataset.cfg_prob = 0.0
    data = {}
    for key, frame_ids in (
        ("left", torch.tensor([0, 1, 2, 3, 4])),
        ("right", torch.tensor([0, 1, 2, 3, 5])),
    ):
        data[f"{key}.latent"] = torch.zeros(2, 3)
        data[f"{key}.latent_num_frames"] = 2
        data[f"{key}.latent_height"] = 1
        data[f"{key}.latent_width"] = 1
        data[f"{key}.frame_ids"] = frame_ids
        data[f"{key}.text_emb"] = torch.zeros(1, 4)

    with pytest.raises(ValueError, match="frame_ids are not aligned"):
        dataset._cat_video_latents(data)


def test_each_camera_latent_grid_must_be_patch_divisible_independently():
    dataset = object.__new__(Dataset)
    dataset.used_video_keys = ["left", "right"]
    dataset.config = SimpleNamespace(patch_size=(1, 2, 2))
    # The concatenated width is divisible by two, but neither camera boundary
    # is. Checking only the total grid would silently change camera addresses.
    data = {
        "left.latent_height": 16,
        "left.latent_width": 15,
        "right.latent_height": 16,
        "right.latent_width": 17,
    }

    with pytest.raises(ValueError, match="each camera WAN latent grid"):
        dataset._camera_wan_grid_shapes(data)


def test_sidecar_rejects_stale_per_camera_partition_with_same_global_grid(
    tmp_path,
):
    dataset = object.__new__(Dataset)
    dataset.used_video_keys = ["left", "right"]
    dataset.config = SimpleNamespace(
        patch_size=(1, 2, 2), rgb_motion_max_tokens=1
    )
    sidecar = tmp_path / "sidecar.pth"
    torch.save(
        {
            "motion_indices": torch.tensor([[0]]),
            "dino_features": torch.ones(1, 1, 4),
            "camera_keys": ["left", "right"],
            "patch_size": (1, 2, 2),
            "spatial_grid_shape": (8, 16),
            "provenance": {
                "camera_wan_grid_shapes": {
                    "left": [8, 8],
                    "right": [8, 8],
                }
            },
        },
        sidecar,
    )
    dataset._resolve_rgb_motion_file = types.MethodType(
        lambda _self, *_args: sidecar, dataset
    )

    with pytest.raises(ValueError, match="per-camera WAN grid mismatch"):
        dataset._load_rgb_motion_sidecar(
            episode_index=0,
            local_start_frame=0,
            local_end_frame=4,
            expected_full_frames=1,
            spatial_tokens_per_frame=128,
            spatial_grid_shape=(8, 16),
            camera_wan_grid_shapes={"left": (8, 7), "right": (8, 9)},
            latent_world_time_ids=torch.tensor([0]),
            latent_frame_ids=torch.tensor([0, 1, 2, 3, 4]),
        )


def test_sidecar_builder_frame_provenance_must_match_wan_latent(tmp_path):
    dataset = object.__new__(Dataset)
    dataset.used_video_keys = ["cam"]
    dataset.config = SimpleNamespace(
        patch_size=(1, 2, 2), rgb_motion_max_tokens=1
    )
    sidecar = tmp_path / "sidecar.pth"
    torch.save(
        {
            "motion_indices": torch.tensor([[0]]),
            "dino_features": torch.ones(1, 1, 4),
            "provenance": {"bundle_frame_ids": [10, 11, 12, 13, 14]},
        },
        sidecar,
    )
    dataset._resolve_rgb_motion_file = types.MethodType(
        lambda _self, *_args: sidecar, dataset
    )

    with pytest.raises(ValueError, match="raw frame alignment mismatch"):
        dataset._load_rgb_motion_sidecar(
            episode_index=0,
            local_start_frame=0,
            local_end_frame=4,
            expected_full_frames=1,
            spatial_tokens_per_frame=1,
            spatial_grid_shape=(1, 1),
            camera_wan_grid_shapes={"cam": (1, 1)},
            latent_world_time_ids=torch.tensor([0]),
            latent_frame_ids=torch.tensor([20, 21, 22, 23, 24]),
        )
