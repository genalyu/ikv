from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from example_client.closed_loop_client import TwamClient


class _FakeTransport:
    def __init__(self) -> None:
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        if request.get("compute_kv_cache"):
            return {"ok": True}
        return {"action": np.zeros((20, 2, 1), dtype=np.float32)}

    def get_server_metadata(self):
        return {}


def _client_and_inputs():
    transport = _FakeTransport()
    client = TwamClient(
        prompt="test instruction",
        num_arms=1,
        cam_names=("cam",),
        tactile_names=("touch",),
        client=transport,
    )
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    state = np.zeros(20, dtype=np.float32)
    return client, transport, {"cam": frame}, {"touch": frame}, state


def _raw_motion_payload():
    camera_key = "observation.images.cam"
    camera = {
        "rgb": np.zeros((1, 2, 3, 3), dtype=np.uint8),
        "depth": np.ones((1, 2, 3), dtype=np.float32),
        "world_from_camera": np.eye(4, dtype=np.float32)[None],
        "intrinsics": np.array(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    }
    return {
        "camera_keys": [camera_key],
        "cameras": {camera_key: camera},
        "anchor_indices": [0],
        "world_time_ids": [7],
        "previous": {
            "camera_keys": [camera_key],
            "cameras": {camera_key: copy.deepcopy(camera)},
        },
    }


def _canonical_motion_payload():
    return {
        "motion_indices": np.array([[2, -1]], dtype=np.int64),
        "motion_valid_mask": np.array([[True, False]]),
        "motion_scores": np.array([[0.9, 0.0]], dtype=np.float32),
        "world_time_id": np.array([[7, -1]], dtype=np.int64),
        "dino_features": np.ones((1, 2, 3), dtype=np.float32),
        "neoforce_features": np.empty((1, 2, 0), dtype=np.float32),
        "observation_flag": np.array([[1, 0]], dtype=np.int64),
        "visual_valid": np.array([[True, False]]),
        "tactile_valid": np.array([[False, False]]),
    }


def test_precomputed_rgb_motion_identity_passthrough_for_infer_and_commit():
    client, transport, cams, tactile, state = _client_and_inputs()
    infer_payload = _canonical_motion_payload()
    commit_payload = _canonical_motion_payload()

    client.infer_chunk(cams, tactile, state, rgb_motion=infer_payload)
    assert transport.requests[-1]["rgb_motion"] is infer_payload

    client.commit_kv_cache(
        [client.pack_images(cams, client.cam_names, "camera")],
        [client.pack_images(tactile, client.tactile_names, "tactile")],
        np.zeros((20, 1, 1), dtype=np.float32),
        state,
        rgb_motion=commit_payload,
    )
    assert transport.requests[-1]["rgb_motion"] is commit_payload


def test_precomputed_and_raw_payloads_are_both_forwarded_without_rewriting():
    client, transport, cams, tactile, state = _client_and_inputs()
    canonical = _canonical_motion_payload()
    raw = _raw_motion_payload()
    canonical_before = copy.deepcopy(canonical)
    raw_before = copy.deepcopy(raw)

    client.infer_chunk(
        cams,
        tactile,
        state,
        rgb_motion=canonical,
        rgb_motion_inputs=raw,
    )

    request = transport.requests[-1]
    assert request["rgb_motion"] is canonical
    assert request["rgb_motion_inputs"] is raw
    for field, value in canonical.items():
        np.testing.assert_array_equal(value, canonical_before[field])
    np.testing.assert_array_equal(
        raw["cameras"]["observation.images.cam"]["depth"],
        raw_before["cameras"]["observation.images.cam"]["depth"],
    )


def test_infer_chunk_forwards_raw_rgb_motion_payload_without_mutation():
    client, transport, cams, tactile, state = _client_and_inputs()
    payload = _raw_motion_payload()
    before = copy.deepcopy(payload)

    client.infer_chunk(
        cams,
        tactile,
        state,
        rgb_motion_inputs=payload,
    )

    request = transport.requests[-1]
    assert request["rgb_motion_inputs"] is payload
    assert payload["camera_keys"] == before["camera_keys"]
    assert payload["world_time_ids"] == before["world_time_ids"]
    for field in ("rgb", "depth", "world_from_camera", "intrinsics"):
        np.testing.assert_array_equal(
            payload["cameras"]["observation.images.cam"][field],
            before["cameras"]["observation.images.cam"][field],
        )


def test_commit_forwards_raw_rgb_motion_payload_without_mutation():
    client, transport, cams, tactile, state = _client_and_inputs()
    payload = _raw_motion_payload()
    before = copy.deepcopy(payload)
    video_keyframes = [client.pack_images(cams, client.cam_names, "camera")]
    tactile_keyframes = [client.pack_images(tactile, client.tactile_names, "tactile")]

    client.commit_kv_cache(
        video_keyframes,
        tactile_keyframes,
        np.zeros((20, 1, 1), dtype=np.float32),
        state,
        rgb_motion_inputs=payload,
    )

    request = transport.requests[-1]
    assert request["rgb_motion_inputs"] is payload
    assert payload["camera_keys"] == before["camera_keys"]
    for field in ("rgb", "depth", "world_from_camera", "intrinsics"):
        np.testing.assert_array_equal(
            payload["cameras"]["observation.images.cam"][field],
            before["cameras"]["observation.images.cam"][field],
        )


def test_legacy_calls_omit_rgb_motion_payload_key():
    client, transport, cams, tactile, state = _client_and_inputs()

    client.infer_chunk(cams, tactile, state)
    assert "rgb_motion" not in transport.requests[-1]
    assert "rgb_motion_inputs" not in transport.requests[-1]

    client.commit_kv_cache(
        [client.pack_images(cams, client.cam_names, "camera")],
        [client.pack_images(tactile, client.tactile_names, "tactile")],
        np.zeros((20, 1, 1), dtype=np.float32),
        state,
    )
    assert "rgb_motion" not in transport.requests[-1]
    assert "rgb_motion_inputs" not in transport.requests[-1]


def test_run_chunk_builds_and_forwards_infer_and_commit_rgb_motion_inputs():
    client, transport, cams, tactile, state = _client_and_inputs()
    infer_payload = _raw_motion_payload()
    commit_payload = _raw_motion_payload()
    observed_cams = {
        "cam": np.full((2, 3, 3), 5, dtype=np.uint8),
    }
    observed_tactile = {
        "touch": np.full((2, 3, 3), 6, dtype=np.uint8),
    }
    builder_calls = []

    def make_infer(current_cams):
        builder_calls.append(("infer", current_cams))
        return infer_payload

    def make_commit(keyframe_cams):
        builder_calls.append(("commit", keyframe_cams))
        return commit_payload

    result = client.run_chunk(
        cams,
        tactile,
        state,
        execute=lambda poses, context: False,
        observe=lambda: (observed_cams, observed_tactile),
        make_infer_rgb_motion_inputs=make_infer,
        make_commit_rgb_motion_inputs=make_commit,
    )

    assert result.committed
    assert len(transport.requests) == 2
    assert transport.requests[0]["rgb_motion_inputs"] is infer_payload
    assert transport.requests[1]["rgb_motion_inputs"] is commit_payload
    assert builder_calls[0] == ("infer", cams)
    assert builder_calls[1][0] == "commit"
    assert builder_calls[1][1] == [observed_cams]


def test_run_chunk_builds_precomputed_sidecars_and_sends_them_with_raw_inputs():
    client, transport, cams, tactile, state = _client_and_inputs()
    infer_canonical = _canonical_motion_payload()
    commit_canonical = _canonical_motion_payload()
    infer_raw = _raw_motion_payload()
    commit_raw = _raw_motion_payload()

    result = client.run_chunk(
        cams,
        tactile,
        state,
        execute=lambda poses, context: False,
        observe=lambda: (cams, tactile),
        make_infer_rgb_motion=lambda current_cams: infer_canonical,
        make_commit_rgb_motion=lambda keyframe_cams: commit_canonical,
        make_infer_rgb_motion_inputs=lambda current_cams: infer_raw,
        make_commit_rgb_motion_inputs=lambda keyframe_cams: commit_raw,
    )

    assert result.committed
    assert transport.requests[0]["rgb_motion"] is infer_canonical
    assert transport.requests[0]["rgb_motion_inputs"] is infer_raw
    assert transport.requests[1]["rgb_motion"] is commit_canonical
    assert transport.requests[1]["rgb_motion_inputs"] is commit_raw


def test_legacy_run_chunk_omits_rgb_motion_payloads():
    client, transport, cams, tactile, state = _client_and_inputs()

    result = client.run_chunk(
        cams,
        tactile,
        state,
        execute=lambda poses, context: False,
        observe=lambda: (cams, tactile),
    )

    assert result.committed
    assert len(transport.requests) == 2
    assert all("rgb_motion" not in request for request in transport.requests)
    assert all("rgb_motion_inputs" not in request for request in transport.requests)


def test_warm_run_chunk_skips_raw_infer_builder_but_calls_canonical_builder():
    client, transport, cams, tactile, state = _client_and_inputs()
    client._cold_chunk = False
    canonical = _canonical_motion_payload()
    calls = {"canonical": 0, "raw": 0}

    def make_canonical(current_cams):
        calls["canonical"] += 1
        return canonical

    def make_raw(current_cams):
        calls["raw"] += 1
        raise AssertionError("warm imagination must not build raw RGB-D input")

    client.run_chunk(
        cams,
        tactile,
        state,
        execute=lambda poses, context: False,
        observe=lambda: (cams, tactile),
        make_infer_rgb_motion=make_canonical,
        make_infer_rgb_motion_inputs=make_raw,
    )

    assert calls == {"canonical": 1, "raw": 0}
    assert transport.requests[0]["rgb_motion"] is canonical
    assert "rgb_motion_inputs" not in transport.requests[0]


def test_run_episode_passes_rgb_motion_builders_to_each_chunk():
    client, transport, cams, tactile, state = _client_and_inputs()
    infer_payload = _raw_motion_payload()
    commit_payload = _raw_motion_payload()
    infer_canonical = _canonical_motion_payload()
    commit_canonical = _canonical_motion_payload()
    counts = {"infer": 0, "commit": 0, "canonical_infer": 0, "canonical_commit": 0}

    def make_infer(current_cams):
        counts["infer"] += 1
        assert current_cams is cams
        return infer_payload

    def make_commit(keyframe_cams):
        counts["commit"] += 1
        assert keyframe_cams == [cams]
        return commit_payload

    def make_canonical_infer(current_cams):
        counts["canonical_infer"] += 1
        return infer_canonical

    def make_canonical_commit(keyframe_cams):
        counts["canonical_commit"] += 1
        return commit_canonical

    chunks = client.run_episode(
        observe=lambda: (cams, tactile),
        get_state=lambda: state,
        execute=lambda poses, context: False,
        max_chunks=1,
        make_infer_rgb_motion=make_canonical_infer,
        make_commit_rgb_motion=make_canonical_commit,
        make_infer_rgb_motion_inputs=make_infer,
        make_commit_rgb_motion_inputs=make_commit,
    )

    assert chunks == 1
    # reset, infer, commit
    assert len(transport.requests) == 3
    assert transport.requests[1]["rgb_motion_inputs"] is infer_payload
    assert transport.requests[2]["rgb_motion_inputs"] is commit_payload
    assert transport.requests[1]["rgb_motion"] is infer_canonical
    assert transport.requests[2]["rgb_motion"] is commit_canonical
    assert counts == {
        "infer": 1,
        "commit": 1,
        "canonical_infer": 1,
        "canonical_commit": 1,
    }
