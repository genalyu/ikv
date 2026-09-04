from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


SCRIPT_PATH = Path(__file__).parents[1] / "script" / "build_rgb_motion_sidecars.py"
SPEC = importlib.util.spec_from_file_location("build_rgb_motion_sidecars", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


class FakeDinoEncoder:
    def __call__(self, rgb: torch.Tensor):
        batch = rgb.shape[0]
        # The camera's constant RGB value becomes its semantic feature and lets
        # the fake detector rank cameras globally after width concatenation.
        if rgb.shape[-1] == 3:
            values = rgb.float().mean(dim=(1, 2, 3))
        else:
            values = rgb.float().mean(dim=(1, 2, 3))
        tokens = values[:, None, None, None].expand(batch, 2, 2, 3).clone()
        return SimpleNamespace(tokens=tokens, grid_size=(2, 2))


class FakeDetector:
    max_tokens = None

    def __init__(self):
        self.calls = 0

    def __call__(self, dino_previous, dino_current, *args, wan_grid_size, **kwargs):
        self.calls += 1
        batch = dino_current.shape[0]
        height, width = tuple(wan_grid_size)
        mask = torch.ones(batch, height, width, dtype=torch.bool)
        camera_score = dino_current.float().mean(dim=(1, 2, 3))
        scores = camera_score[:, None, None].expand(batch, height, width).clone()
        return SimpleNamespace(wan_motion_mask=mask, wan_motion_score=scores)


def _write_fixture(tmp_path: Path, *, camera_order=("cam_a", "cam_b")) -> Path:
    frame_ids = [10, 11, 12, 13, 14]
    episode_name = "episode_000007_0_5.pth"
    camera_payloads = {}
    latent_files = {}
    for camera_index, camera_key in enumerate(("cam_a", "cam_b"), start=1):
        latent_path = (
            tmp_path / "latents" / "chunk-000" / camera_key / episode_name
        )
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_num_frames": 2,
                "latent_height": 2,
                "latent_width": 4,
                "video_num_frames": len(frame_ids),
                "frame_ids": frame_ids,
                "temporal_provenance": {
                    "schema_version": 1,
                    "anchor_semantics": "causal_chunk_end",
                    "temporal_stride": 4,
                    "latent_anchor_indices": [0, 4],
                    "latent_anchor_frame_ids": [10, 14],
                },
            },
            latent_path,
        )
        latent_files[camera_key] = str(latent_path.relative_to(tmp_path))
        camera_payloads[camera_key] = {
            "rgb": torch.full(
                (len(frame_ids), 4, 4, 3), 10 * camera_index, dtype=torch.uint8
            ),
            "depth": torch.ones(len(frame_ids), 4, 4),
            "camera_pose": torch.eye(4).expand(len(frame_ids), -1, -1).clone(),
            "camera_intrinsics": torch.tensor(
                [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]
            ),
        }

    ordered_cameras = {key: camera_payloads[key] for key in camera_order}
    bundle_path = tmp_path / "bundles" / "episode_000007_0_5.pth"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"frame_ids": frame_ids, "cameras": ordered_cameras}, bundle_path)

    manifest = {
        "schema_version": 1,
        "camera_keys": ["cam_a", "cam_b"],
        "output_root": "rgb_motion",
        "dino_model": "locally-cached-dinov2",
        "max_tokens": 2,
        "patch_size": [1, 2, 2],
        "first_frame_policy": "empty",
        "segments": [
            {
                "episode_index": 7,
                "chunk_index": 0,
                "start_frame": 0,
                "end_frame": 5,
                "bundle": str(bundle_path.relative_to(tmp_path)),
                "latent_files": latent_files,
                "anchor_indices": [0, 4],
                "world_time_ids": [30, 31],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _append_fixture_segment(
    manifest_path: Path,
    *,
    episode_index: int,
    chunk_index: int,
) -> Path:
    """Add another valid segment by copying the fixture's tensor payloads."""

    root = manifest_path.parent
    manifest = _read_manifest(manifest_path)
    source = manifest["segments"][0]
    start = source["start_frame"]
    end = source["end_frame"]
    filename = f"episode_{episode_index:06d}_{start}_{end}.pth"

    source_bundle = root / source["bundle"]
    bundle_path = root / "bundles" / filename
    torch.save(
        torch.load(source_bundle, map_location="cpu", weights_only=False),
        bundle_path,
    )

    latent_files = {}
    for camera_key in manifest["camera_keys"]:
        source_latent = root / source["latent_files"][camera_key]
        latent_path = (
            root
            / "latents"
            / f"chunk-{chunk_index:03d}"
            / camera_key
            / filename
        )
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.load(source_latent, map_location="cpu", weights_only=False),
            latent_path,
        )
        latent_files[camera_key] = str(latent_path.relative_to(root))

    manifest["segments"].append(
        {
            "episode_index": episode_index,
            "chunk_index": chunk_index,
            "start_frame": start,
            "end_frame": end,
            "bundle": str(bundle_path.relative_to(root)),
            "latent_files": latent_files,
            "anchor_indices": list(source["anchor_indices"]),
            "world_time_ids": [40, 41],
        }
    )
    _write_manifest(manifest_path, manifest)
    return (
        root
        / "rgb_motion"
        / f"chunk-{chunk_index:03d}"
        / filename
    )


def test_builder_writes_dataset_ready_canonical_sidecar(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    detector = FakeDetector()

    written = BUILDER.build_rgb_motion_sidecars(
        manifest_path,
        dino_encoder=FakeDinoEncoder(),
        detector=detector,
    )

    expected = tmp_path / "rgb_motion" / "chunk-000" / "episode_000007_0_5.pth"
    assert written == [expected]
    assert detector.calls == 2  # one transition for each camera
    payload = torch.load(expected, map_location="cpu", weights_only=False)

    assert all(field in payload for field in BUILDER.CANONICAL_FIELDS)
    assert payload["motion_indices"].shape == (2, 2)
    assert payload["motion_indices"].tolist() == [[-1, -1], [2, 3]]
    assert payload["motion_valid_mask"].tolist() == [
        [False, False],
        [True, True],
    ]
    assert payload["world_time_id"].tolist() == [[-1, -1], [31, 31]]
    assert payload["observation_flag"].tolist() == [[0, 0], [1, 1]]
    assert payload["visual_valid"].tolist() == payload["motion_valid_mask"].tolist()
    assert not payload["tactile_valid"].any()
    assert payload["neoforce_features"].shape == (2, 2, 0)
    assert payload["camera_keys"] == ["cam_a", "cam_b"]
    assert payload["patch_size"] == (1, 2, 2)
    assert payload["spatial_grid_shape"] == (1, 4)
    assert payload["latent_num_frames"] == 2
    assert payload["provenance"]["bundle_frame_ids"] == [10, 11, 12, 13, 14]
    assert payload["provenance"]["world_time_ids"] == [30, 31]
    assert all(
        not isinstance(value, torch.Tensor) or value.device.type == "cpu"
        for value in payload.values()
    )


def test_camera_order_is_a_hard_alignment_contract(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path, camera_order=("cam_b", "cam_a"))

    with pytest.raises(ValueError, match="camera order mismatch"):
        BUILDER.build_rgb_motion_sidecars(
            manifest_path,
            dino_encoder=FakeDinoEncoder(),
            detector=FakeDetector(),
        )


def test_latent_frame_ids_must_match_raw_bundle(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    latent_path = tmp_path / manifest["segments"][0]["latent_files"]["cam_b"]
    latent = torch.load(latent_path, map_location="cpu", weights_only=False)
    latent["frame_ids"] = [10, 11, 12, 13, 99]
    torch.save(latent, latent_path)

    with pytest.raises(ValueError, match="not aligned"):
        BUILDER.build_rgb_motion_sidecars(
            manifest_path,
            dino_encoder=FakeDinoEncoder(),
            detector=FakeDetector(),
        )


def test_anchor_count_must_equal_latent_time_count(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    manifest["segments"][0]["anchor_indices"] = [0]
    manifest["segments"][0]["world_time_ids"] = [30]
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="latent frame count 2.*1 explicit anchors"):
        BUILDER.build_rgb_motion_sidecars(
            manifest_path,
            dino_encoder=FakeDinoEncoder(),
            detector=FakeDetector(),
        )


def test_manifest_anchors_must_match_latent_temporal_provenance(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    # Same count and still in range, so the old length-only check accepted it.
    manifest["segments"][0]["anchor_indices"] = [1, 4]
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="disagrees with latent temporal provenance"):
        BUILDER.build_rgb_motion_sidecars(
            manifest_path,
            dino_encoder=FakeDinoEncoder(),
            detector=FakeDetector(),
        )


def test_latent_without_temporal_provenance_is_rejected(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    latent_path = tmp_path / manifest["segments"][0]["latent_files"]["cam_a"]
    latent = torch.load(latent_path, map_location="cpu", weights_only=False)
    latent.pop("temporal_provenance")
    torch.save(latent, latent_path)

    with pytest.raises(KeyError, match="missing temporal_provenance.*Re-encode"):
        BUILDER.build_rgb_motion_sidecars(
            manifest_path,
            dino_encoder=FakeDinoEncoder(),
            detector=FakeDetector(),
        )


def test_duplicate_output_is_rejected_before_processing(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    manifest["segments"].append(dict(manifest["segments"][0]))
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="duplicate RGB-motion output"):
        BUILDER.build_rgb_motion_sidecars(
            manifest_path,
            dino_encoder=FakeDinoEncoder(),
            detector=FakeDetector(),
        )


def test_all_existing_outputs_skip_without_loading_dino_checkpoint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    manifest.pop("dino_model")
    _write_manifest(manifest_path, manifest)
    destination = (
        tmp_path / "rgb_motion" / "chunk-000" / "episode_000007_0_5.pth"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"already_complete": True}, destination)

    dino_calls = []

    def unexpected_dino_load(*args, **kwargs):
        dino_calls.append((args, kwargs))
        raise AssertionError("DINO must not load when every output is skipped")

    monkeypatch.setattr(
        BUILDER.FrozenDinoV2PatchEncoder,
        "from_pretrained",
        unexpected_dino_load,
    )
    written = BUILDER.main(
        ["--manifest", str(manifest_path), "--device", "cpu"],
        detector=FakeDetector(),
    )

    assert written == []
    assert dino_calls == []
    assert f"skip existing {destination}" in capsys.readouterr().out


def test_partial_existing_outputs_process_only_missing_segments(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    missing_destination = _append_fixture_segment(
        manifest_path,
        episode_index=8,
        chunk_index=1,
    )
    manifest = _read_manifest(manifest_path)
    existing_destination = (
        tmp_path / "rgb_motion" / "chunk-000" / "episode_000007_0_5.pth"
    )
    existing_destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"already_complete": True}, existing_destination)

    # If the skipped segment were processed, either invalid payload would make
    # the build fail.  Manifest path syntax remains valid for preflight.
    first = manifest["segments"][0]
    torch.save("invalid skipped bundle", tmp_path / first["bundle"])
    for path in first["latent_files"].values():
        torch.save("invalid skipped latent", tmp_path / path)

    detector = FakeDetector()
    written = BUILDER.build_rgb_motion_sidecars(
        manifest_path,
        dino_encoder=FakeDinoEncoder(),
        detector=detector,
    )

    assert written == [missing_destination]
    assert detector.calls == 2
    assert torch.load(
        existing_destination, map_location="cpu", weights_only=False
    ) == {"already_complete": True}


def test_latent_chunk_mismatch_fails_before_dino_loading(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    original = manifest["segments"][0]["latent_files"]["cam_a"]
    manifest["segments"][0]["latent_files"]["cam_a"] = original.replace(
        "chunk-000", "chunk-001"
    )
    _write_manifest(manifest_path, manifest)
    dino_calls = []

    def unexpected_dino_load(*args, **kwargs):
        dino_calls.append((args, kwargs))
        raise AssertionError("chunk validation must run before DINO loading")

    monkeypatch.setattr(
        BUILDER.FrozenDinoV2PatchEncoder,
        "from_pretrained",
        unexpected_dino_load,
    )
    with pytest.raises(
        ValueError,
        match=r"resolves under 'chunk-001'.*chunk_index=0.*'chunk-000'",
    ):
        BUILDER.main(["--manifest", str(manifest_path), "--device", "cpu"])

    assert dino_calls == []


def test_latent_path_without_standard_chunk_directory_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    filename = Path(
        manifest["segments"][0]["latent_files"]["cam_a"]
    ).name
    manifest["segments"][0]["latent_files"]["cam_a"] = (
        f"latents/cam_a/{filename}"
    )
    _write_manifest(manifest_path, manifest)
    dino_calls = []

    def unexpected_dino_load(*args, **kwargs):
        dino_calls.append((args, kwargs))
        raise AssertionError("chunk validation must run before DINO loading")

    monkeypatch.setattr(
        BUILDER.FrozenDinoV2PatchEncoder,
        "from_pretrained",
        unexpected_dino_load,
    )
    with pytest.raises(ValueError, match=r"paths without a chunk-NNN directory"):
        BUILDER.main(["--manifest", str(manifest_path), "--device", "cpu"])

    assert dino_calls == []


@pytest.mark.parametrize(
    ("extra_args", "expected_local_only"),
    [([], True), (["--allow-dino-download"], False)],
)
def test_cli_dino_loading_is_local_only_unless_explicitly_enabled(
    tmp_path: Path, monkeypatch, extra_args, expected_local_only
) -> None:
    manifest_path = _write_fixture(tmp_path)
    calls = []

    def fake_from_pretrained(source, **kwargs):
        calls.append((source, kwargs))
        return FakeDinoEncoder()

    monkeypatch.setattr(
        BUILDER.FrozenDinoV2PatchEncoder,
        "from_pretrained",
        fake_from_pretrained,
    )
    BUILDER.main(
        ["--manifest", str(manifest_path), "--device", "cpu", *extra_args],
        detector=FakeDetector(),
    )

    assert len(calls) == 1
    assert calls[0][0] == "locally-cached-dinov2"
    assert calls[0][1]["local_files_only"] is expected_local_only


def test_uint16_depth_is_not_silently_rescaled(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = _read_manifest(manifest_path)
    bundle_path = tmp_path / manifest["segments"][0]["bundle"]
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    bundle["cameras"]["cam_a"]["depth"] = torch.ones(5, 4, 4, dtype=torch.uint16)
    torch.save(bundle, bundle_path)

    with pytest.raises(TypeError, match="will not guess a uint16 scale"):
        BUILDER.build_rgb_motion_sidecars(
            manifest_path,
            dino_encoder=FakeDinoEncoder(),
            detector=FakeDetector(),
        )
