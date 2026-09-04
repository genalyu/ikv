"""CPU-only regression tests for the RGB-motion serving boundary.

The real server owns CUDA models and optional serving dependencies.  These
tests load only its class definition with tiny import stubs, then construct an
instance through ``__new__``.  That keeps the contract between the online
sidecar, CFG batching, and sparse WAN output reconstruction testable without a
checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from n0_twam.models.rgb_motion import (
    SparsePatchGather,
    SparsePatchScatter,
    patchify_latents,
)


def _load_server_class():
    module_path = Path(__file__).parents[1] / "n0_twam" / "n0_twam_server.py"

    configs = types.ModuleType("configs")
    configs.TWAM_CONFIGS = {}

    distributed = types.ModuleType("distributed")
    distributed.__path__ = []
    distributed_fsdp = types.ModuleType("distributed.fsdp")
    distributed_fsdp.shard_model = lambda *args, **kwargs: None
    distributed_util = types.ModuleType("distributed.util")
    distributed_util._configure_model = lambda model, **kwargs: model
    distributed_util.init_distributed = lambda *args, **kwargs: None

    models = types.ModuleType("models")
    models.__path__ = []
    models_utils = types.ModuleType("models.utils")
    for name in (
        "WanVAEStreamingWrapper",
        "load_text_encoder",
        "load_tokenizer",
        "load_transformer",
        "load_vae",
    ):
        setattr(models_utils, name, object)
    models_rgb_motion = types.ModuleType("models.rgb_motion")
    models_rgb_motion.SparsePatchGather = SparsePatchGather
    models_rgb_motion.SparsePatchScatter = SparsePatchScatter

    utils = types.ModuleType("utils")
    utils.FlowMatchScheduler = object
    utils.data_seq_to_patch = lambda *args, **kwargs: None
    utils.get_mesh_id = lambda *args, **kwargs: torch.empty(4, 0)
    utils.init_logger = lambda *args, **kwargs: None
    utils.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    utils.run_async_server_mode = lambda *args, **kwargs: None
    utils.save_async = lambda *args, **kwargs: None

    # Avoid importing the heavyweight Diffusers serving surface.  None of these
    # objects participates in the helpers under test.
    diffusers = types.ModuleType("diffusers")
    diffusers.__path__ = []
    diffusers_video = types.ModuleType("diffusers.video_processor")
    diffusers_video.VideoProcessor = object
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.export_to_video = lambda *args, **kwargs: None
    diffusers_pipelines = types.ModuleType("diffusers.pipelines")
    diffusers_pipelines.__path__ = []
    diffusers_wan = types.ModuleType("diffusers.pipelines.wan")
    diffusers_wan.__path__ = []
    diffusers_pipeline_wan = types.ModuleType("diffusers.pipelines.wan.pipeline_wan")
    diffusers_pipeline_wan.prompt_clean = lambda value: value

    stubs = {
        "configs": configs,
        "distributed": distributed,
        "distributed.fsdp": distributed_fsdp,
        "distributed.util": distributed_util,
        "models": models,
        "models.utils": models_utils,
        "models.rgb_motion": models_rgb_motion,
        "utils": utils,
        "diffusers": diffusers,
        "diffusers.video_processor": diffusers_video,
        "diffusers.utils": diffusers_utils,
        "diffusers.pipelines": diffusers_pipelines,
        "diffusers.pipelines.wan": diffusers_wan,
        "diffusers.pipelines.wan.pipeline_wan": diffusers_pipeline_wan,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(
            "_rgb_motion_server_under_test", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TWAM_Server
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


TWAMServer = _load_server_class()


class _TestStreamingVAE:
    def __init__(self, *, temporal_stride=4, warm=False):
        self.vae = SimpleNamespace(
            config=SimpleNamespace(scale_factor_temporal=temporal_stride)
        )
        self.feat_cache = [torch.tensor([1.0]) if warm else None]

    def clear_cache(self):
        self.feat_cache = [None]


def _server(
    *,
    enabled: bool = True,
    patch_size=(1, 2, 2),
    online: bool = False,
    camera_keys=("camera",),
    first_frame_policy="require_previous",
):
    server = TWAMServer.__new__(TWAMServer)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.job_config = SimpleNamespace(
        use_rgb_motion_tokens=enabled,
        patch_size=patch_size,
        rgb_motion_online_preprocess=online,
        rgb_motion_max_tokens=3,
        rgb_motion_first_frame_policy=first_frame_policy,
        obs_cam_keys=list(camera_keys),
        # 64x64 -> WAN latent 4x4 -> transformer patch grid 2x2, so the
        # canonical test addresses 1 and 3 are both real cells.
        height=64,
        width=64,
    )
    server._last_rgb_motion = None
    server._rgb_motion_preprocessor = None
    server._rgb_motion_dino_encoder = None
    server._rgb_motion_previous_raw_frames = None
    server.streaming_vae = _TestStreamingVAE()
    server.vae = server.streaming_vae.vae
    server.rgb_patch_gather = SparsePatchGather(patch_size)
    server.rgb_patch_scatter = SparsePatchScatter()
    return server


def _one_frame_payload():
    return {
        "motion_indices": torch.tensor([[1, -1, 3]]),
        "motion_valid_mask": torch.tensor([[True, False, True]]),
        "motion_scores": torch.tensor([[0.8, 0.0, 0.4]]),
        "dino_features": torch.tensor(
            [[[1.0, 0.0], [9.0, 9.0], [0.0, 1.0]]]
        ),
        # RGB-only is represented by an explicit zero-width NeoForce feature.
        "neoforce_features": torch.empty(1, 3, 0),
        "visual_valid": torch.tensor([[True, False, True]]),
        "tactile_valid": torch.zeros(1, 3, dtype=torch.bool),
        # The server is authoritative for time/source and must overwrite these.
        "world_time_id": torch.full((1, 3), 999),
        "observation_flag": torch.zeros(1, 3, dtype=torch.long),
    }


def _payload_for_frames(num_frames):
    payload = _one_frame_payload()
    return {
        name: value.repeat(num_frames, *([1] * (value.dim() - 1)))
        for name, value in payload.items()
    }


def _raw_camera(num_frames=1, value=0):
    rgb = torch.full(
        (num_frames, 8, 8, 3), value, dtype=torch.uint8
    )
    depth = torch.ones(num_frames, 8, 8)
    return {
        "rgb": rgb,
        "depth": depth,
        "world_from_camera": torch.eye(4).repeat(num_frames, 1, 1),
        "intrinsics": torch.tensor(
            [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]
        ),
    }


def _raw_inputs(
    camera_keys=("camera",),
    *,
    num_frames=1,
    previous=False,
    include_anchor=True,
):
    payload = {
        "camera_keys": list(camera_keys),
        "cameras": {
            key: _raw_camera(num_frames, value=index)
            for index, key in enumerate(camera_keys)
        },
    }
    if include_anchor:
        payload["anchor_indices"] = list(range(num_frames))
    if previous:
        payload["previous"] = {
            "camera_keys": list(camera_keys),
            "cameras": {
                key: _raw_camera(1, value=10 + index)
                for index, key in enumerate(camera_keys)
            },
        }
    return payload


def _observation_with_raw(raw):
    camera_keys = list(raw["cameras"])
    first_rgb = raw["cameras"][camera_keys[0]]["rgb"]
    num_frames = first_rgb.shape[0] if first_rgb.ndim == 4 else 1
    observations = []
    for frame_index in range(num_frames):
        frame = {}
        for camera_key in camera_keys:
            rgb = raw["cameras"][camera_key]["rgb"]
            if rgb.ndim == 3:
                image = rgb
            else:
                image = rgb[frame_index]
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = image.permute(1, 2, 0)
            frame[camera_key] = image.detach().cpu().numpy().copy()
        observations.append(frame)
    return {"obs": observations, "rgb_motion_inputs": raw}


class _RecordingRawPreprocessor:
    def __init__(self):
        self.calls = []

    def __call__(self, cameras, **kwargs):
        self.calls.append((cameras, kwargs))
        num_frames = len(kwargs["anchor_indices"])
        payload = _payload_for_frames(num_frames)
        payload.update({
            "camera_keys": list(cameras),
            "patch_size": (1, 2, 2),
            # Test servers use 64x64 input -> 4x4 latent -> 2x2 per camera.
            "spatial_grid_shape": (2, 2 * len(cameras)),
        })
        return payload


def test_rgb_motion_disabled_has_no_sidecar_requirement():
    server = _server(enabled=False)
    assert server._rgb_motion_for_frames({}, 3, 7, observed=False) is None


def test_precomputed_sidecar_takes_precedence_over_online_raw_inputs():
    server = _server(online=True)

    class _MustNotRun:
        def __call__(self, *args, **kwargs):
            raise AssertionError("raw preprocessor ran despite precomputed sidecar")

    server._rgb_motion_preprocessor = _MustNotRun()
    sidecar = server._rgb_motion_for_frames(
        {
            "rgb_motion": _one_frame_payload(),
            "rgb_motion_inputs": _raw_inputs(previous=True),
        },
        1,
        3,
        observed=True,
    )

    assert sidecar["world_time_id"][0, 0, 0].item() == 3
    assert server._rgb_motion_previous_raw_frames is None


def test_precomputed_sidecar_preparation_does_not_require_raw_vae_anchor_state():
    server = _server(online=True)
    # A canonical sidecar already addresses latent rows. It must not inspect
    # raw causal anchors or even require the streaming cache introspection hook.
    server.streaming_vae = object()
    prepared = server._prepare_observed_rgb_motion(
        {"rgb_motion": _one_frame_payload()}, frame_st_id=4
    )

    assert prepared["num_frames"] == 1
    assert prepared["sidecar"]["world_time_id"][0, 0, 0].item() == 4


def test_multicamera_precomputed_sidecar_requires_exact_grid_provenance():
    camera_keys = ("left", "right")
    server = _server(camera_keys=camera_keys)

    with pytest.raises(KeyError, match="multi-camera.*provenance"):
        server._rgb_motion_for_frames(
            {"rgb_motion": _one_frame_payload()}, 1, 0, observed=True
        )

    payload = _one_frame_payload()
    payload.update({
        "camera_keys": list(reversed(camera_keys)),
        "patch_size": (1, 2, 2),
        "spatial_grid_shape": (2, 4),
    })
    with pytest.raises(ValueError, match="width-concat order"):
        server._rgb_motion_for_frames(
            {"rgb_motion": payload}, 1, 0, observed=True
        )

    payload["camera_keys"] = list(camera_keys)
    payload["patch_size"] = (1, 4, 2)
    with pytest.raises(ValueError, match="patch_size does not match"):
        server._rgb_motion_for_frames(
            {"rgb_motion": payload}, 1, 0, observed=True
        )

    payload["patch_size"] = (1, 2, 2)
    payload["spatial_grid_shape"] = (1, 99)
    with pytest.raises(ValueError, match="spatial_grid_shape does not match"):
        server._rgb_motion_for_frames(
            {"rgb_motion": payload}, 1, 0, observed=True
        )

    payload["spatial_grid_shape"] = (2, 4)
    result = server._rgb_motion_for_frames(
        {"rgb_motion": payload}, 1, 0, observed=True
    )
    assert result["motion_indices"].shape == (1, 1, 3)


def test_single_camera_precomputed_minimal_schema_remains_compatible():
    server = _server(camera_keys=("camera",))
    result = server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 0, observed=True
    )
    assert result["motion_indices"].shape == (1, 1, 3)


def test_online_raw_rgbd_generates_observed_sidecar_and_reuses_previous_state():
    camera_keys = ("left", "right")
    server = _server(online=True, camera_keys=camera_keys)
    processor = _RecordingRawPreprocessor()
    server._rgb_motion_preprocessor = processor

    first = server._rgb_motion_for_frames(
        _observation_with_raw(_raw_inputs(camera_keys, previous=True)),
        1,
        7,
        observed=True,
    )

    assert tuple(processor.calls[0][0]) == camera_keys
    assert processor.calls[0][1]["anchor_indices"].tolist() == [0]
    assert processor.calls[0][1]["world_time_ids"].tolist() == [7]
    assert processor.calls[0][1]["previous_frames"] is not None
    assert first["world_time_id"].tolist() == [[[7, -1, 7]]]
    assert set(server._rgb_motion_previous_raw_frames) == set(camera_keys)
    assert set(server._rgb_motion_previous_raw_frames["left"]) == {
        "rgb",
        "depth",
        "camera_pose",
        "camera_intrinsics",
    }

    # The second raw observation has no explicit previous block.  The server
    # supplies the last successfully processed raw frame from the first call.
    second = server._rgb_motion_for_frames(
        _observation_with_raw(_raw_inputs(camera_keys)),
        1,
        8,
        observed=True,
    )
    cached_previous = processor.calls[1][1]["previous_frames"]
    assert cached_previous is not None
    assert cached_previous["right"]["rgb"].shape == (8, 8, 3)
    assert second["world_time_id"][0, 0, 0].item() == 8
    # Provenance was checked on ingestion; compact internal carry-forward keeps
    # only the canonical nine fields and must remain usable for prediction.
    predicted = server._rgb_motion_for_frames({}, 2, 9, observed=False)
    assert predicted["world_time_id"][:, :, 0].tolist() == [[9, 10]]


def test_online_previous_state_uses_last_causal_warm_anchor():
    server = _server(online=True, first_frame_policy="empty")
    server.streaming_vae.feat_cache = [torch.tensor([1.0])]
    processor = _RecordingRawPreprocessor()
    server._rgb_motion_preprocessor = processor
    raw = _raw_inputs(num_frames=8)
    raw["anchor_indices"] = [3, 7]
    camera = raw["cameras"]["camera"]
    for frame_index in range(8):
        camera["rgb"][frame_index].fill_(frame_index)
        camera["depth"][frame_index].fill_(10 + frame_index)
    camera["world_from_camera"][:, 0, 3] = torch.arange(8).float()

    server._rgb_motion_for_frames(
        _observation_with_raw(raw), 2, 20, observed=True
    )

    cached = server._rgb_motion_previous_raw_frames["camera"]
    assert processor.calls[0][1]["anchor_indices"].tolist() == [3, 7]
    assert cached["rgb"].eq(7).all()
    assert cached["depth"].eq(17).all()
    assert cached["camera_pose"][0, 3].item() == 7


def test_online_raw_rgb_must_equal_every_wan_vae_observation_frame():
    server = _server(online=True)
    server.streaming_vae.feat_cache = [torch.tensor([1.0])]
    server._rgb_motion_preprocessor = _RecordingRawPreprocessor()
    raw = _raw_inputs(num_frames=4, previous=True)
    raw["anchor_indices"] = [3]
    observation = _observation_with_raw(raw)
    # Frame 1 is not selected by the sole causal-end anchor, but it is encoded by
    # the streaming WAN VAE and therefore cannot come from a second RGB source.
    observation["obs"][1]["camera"][0, 0, 0] = 7

    with pytest.raises(ValueError, match="does not match.*WAN VAE"):
        server._rgb_motion_for_frames(
            observation, 1, 0, observed=True
        )


def test_online_rgb_binding_allows_dtype_only_difference_but_not_length_or_keys():
    server = _server(online=True, first_frame_policy="empty")
    server._rgb_motion_preprocessor = _RecordingRawPreprocessor()
    raw = _raw_inputs()
    observation = _observation_with_raw(raw)
    observation["obs"][0]["camera"] = observation["obs"][0]["camera"].astype(
        "float32"
    )
    server._rgb_motion_for_frames(observation, 1, 0, observed=True)

    wrong_length = _observation_with_raw(_raw_inputs(num_frames=2))
    wrong_length["obs"] = wrong_length["obs"][:1]
    with pytest.raises(ValueError, match="raw T=2, obs T=1"):
        server._rgb_motion_for_frames(
            wrong_length, 2, 0, observed=True
        )

    missing_camera = _observation_with_raw(_raw_inputs())
    missing_camera["obs"][0].pop("camera")
    with pytest.raises(KeyError, match="missing configured RGB cameras"):
        server._rgb_motion_for_frames(
            missing_camera, 1, 0, observed=True
        )


def test_online_rgb_preflight_fails_before_prediction_cache_is_cleared():
    server = _server(online=True, first_frame_policy="empty")

    class _Transformer:
        cleared = False

        def clear_pred_cache(self, name):
            self.cleared = True

    server.transformer = _Transformer()
    raw = _raw_inputs()
    observation = _observation_with_raw(raw)
    observation["obs"][0]["camera"][0, 0, 0] = 9

    with pytest.raises(ValueError, match="does not match.*WAN VAE"):
        server._compute_kv_cache(observation)
    assert not server.transformer.cleared


def test_grounding_canonical_error_preserves_cache_vae_and_rgb_episode_state():
    server = _server()
    server.cache_name = "pos"
    server.frame_st_id = 5
    old_last = {"sentinel": torch.tensor(1)}
    old_previous = {"camera": {"sentinel": torch.tensor(2)}}
    server._last_rgb_motion = old_last
    server._rgb_motion_previous_raw_frames = old_previous
    calls = {"clear": 0, "encode": 0}

    class _Transformer:
        def clear_pred_cache(self, _name):
            calls["clear"] += 1

    server.transformer = _Transformer()
    server._encode_obs = lambda _obs: calls.__setitem__(
        "encode", calls["encode"] + 1
    )
    payload = _one_frame_payload()
    payload["motion_indices"] = torch.tensor([[1, -1, 1]])

    with pytest.raises(ValueError, match="duplicate valid addresses"):
        server._compute_kv_cache({"obs": [], "rgb_motion": payload})

    assert calls == {"clear": 0, "encode": 0}
    assert server._last_rgb_motion is old_last
    assert server._rgb_motion_previous_raw_frames is old_previous


def test_grounding_rejects_out_of_grid_address_before_cache_or_vae_mutation():
    server = _server()
    server.cache_name = "pos"
    server.frame_st_id = 5
    calls = {"clear": 0, "encode": 0}

    class _Transformer:
        def clear_pred_cache(self, _name):
            calls["clear"] += 1

    server.transformer = _Transformer()
    server._encode_obs = lambda _obs: calls.__setitem__(
        "encode", calls["encode"] + 1
    )
    payload = _one_frame_payload()
    payload["motion_indices"][0, 0] = 4  # 64x64 config -> 2x2 grid -> [0,4)

    with pytest.raises(IndexError, match="outside.*WAN patch grid"):
        server._compute_kv_cache({"obs": [], "rgb_motion": payload})

    assert calls == {"clear": 0, "encode": 0}


def test_grounding_rejects_live_cache_feature_width_drift_before_mutation():
    server = _server()
    server.cache_name = "pos"
    server.frame_st_id = 5
    calls = {"clear": 0, "encode": 0}

    class _Transformer:
        def get_semantic_cache(self, *_args, **_kwargs):
            return {
                "dino": torch.zeros(8, 2),
                "neoforce": torch.empty(8, 0),
            }

        def clear_pred_cache(self, _name):
            calls["clear"] += 1

    server.transformer = _Transformer()
    server._encode_obs = lambda _obs: calls.__setitem__(
        "encode", calls["encode"] + 1
    )
    payload = _one_frame_payload()
    payload["dino_features"] = torch.zeros(1, 3, 3)

    with pytest.raises(ValueError, match="feature dimensions changed"):
        server._compute_kv_cache({"obs": [], "rgb_motion": payload})

    assert calls == {"clear": 0, "encode": 0}


def test_grounding_raw_dino_error_runs_before_cache_or_vae_and_restores_state():
    server = _server(online=True, first_frame_policy="empty")
    server.cache_name = "pos"
    server.frame_st_id = 5
    old_last = {"sentinel": torch.tensor(1)}
    old_previous = {"camera": {"sentinel": torch.tensor(2)}}
    server._last_rgb_motion = old_last
    server._rgb_motion_previous_raw_frames = old_previous
    calls = {"producer": 0, "clear": 0, "encode": 0}

    class _FailingDinoProducer:
        def __call__(self, *_args, **_kwargs):
            calls["producer"] += 1
            raise RuntimeError("DINO checkpoint unavailable")

    class _Transformer:
        def clear_pred_cache(self, _name):
            calls["clear"] += 1

    server._rgb_motion_preprocessor = _FailingDinoProducer()
    server.transformer = _Transformer()
    server._encode_obs = lambda _obs: calls.__setitem__(
        "encode", calls["encode"] + 1
    )
    observation = _observation_with_raw(_raw_inputs())

    with pytest.raises(RuntimeError, match="DINO checkpoint unavailable"):
        server._compute_kv_cache(observation)

    assert calls == {"producer": 1, "clear": 0, "encode": 0}
    assert server._last_rgb_motion is old_last
    assert server._rgb_motion_previous_raw_frames is old_previous


def test_grounding_bad_raw_anchor_fails_before_producer_cache_or_vae_mutation():
    server = _server(online=True, first_frame_policy="empty")
    server.cache_name = "pos"
    server.frame_st_id = 5
    old_last = {"sentinel": torch.tensor(1)}
    old_previous = {"camera": {"sentinel": torch.tensor(2)}}
    server._last_rgb_motion = old_last
    server._rgb_motion_previous_raw_frames = old_previous
    calls = {"producer": 0, "clear": 0, "encode": 0}

    class _Producer:
        def __call__(self, *_args, **_kwargs):
            calls["producer"] += 1
            raise AssertionError("bad anchors must be rejected before DINO")

    class _Transformer:
        def clear_pred_cache(self, _name):
            calls["clear"] += 1

    server._rgb_motion_preprocessor = _Producer()
    server.transformer = _Transformer()
    server._encode_obs = lambda _obs: calls.__setitem__(
        "encode", calls["encode"] + 1
    )
    raw = _raw_inputs()
    raw["anchor_indices"] = [1]  # one raw row exists, so only address 0 is legal
    observation = _observation_with_raw(raw)

    with pytest.raises(IndexError, match=r"raw RGB sequence \[0,1\)"):
        server._compute_kv_cache(observation)

    assert calls == {"producer": 0, "clear": 0, "encode": 0}
    assert server._last_rgb_motion is old_last
    assert server._rgb_motion_previous_raw_frames is old_previous


def test_cold_infer_canonical_error_does_not_advance_any_encoder_state():
    server = _server()
    server.job_config.frame_chunk_size = 2
    old_last = {"sentinel": torch.tensor(1)}
    old_previous = {"camera": {"sentinel": torch.tensor(2)}}
    old_init = torch.tensor(3)
    old_tactile = {"sentinel": torch.tensor(4)}
    old_background = torch.tensor(5)
    server._last_rgb_motion = old_last
    server._rgb_motion_previous_raw_frames = old_previous
    server.init_latent = old_init
    server.last_tactile_latents = old_tactile
    server._last_observed_video_latent = old_background
    calls = {"video": 0, "tactile": 0}
    server._encode_obs = lambda _obs: calls.__setitem__(
        "video", calls["video"] + 1
    )
    server._encode_tactile_obs = lambda _obs: calls.__setitem__(
        "tactile", calls["tactile"] + 1
    )
    payload = _one_frame_payload()
    payload["motion_indices"] = torch.tensor([[1, -1, 1]])

    with pytest.raises(ValueError, match="duplicate valid addresses"):
        server._infer({"rgb_motion": payload}, frame_st_id=0)

    assert calls == {"video": 0, "tactile": 0}
    assert server._last_rgb_motion is old_last
    assert server._rgb_motion_previous_raw_frames is old_previous
    assert server.init_latent is old_init
    assert server.last_tactile_latents is old_tactile
    assert server._last_observed_video_latent is old_background


def test_observed_rgb_preparation_is_single_pass_and_commits_only_explicitly():
    server = _server(online=True, first_frame_policy="empty")
    processor = _RecordingRawPreprocessor()
    server._rgb_motion_preprocessor = processor
    old_last = {"sentinel": torch.tensor(1)}
    old_previous = {"camera": {"sentinel": torch.tensor(2)}}
    server._last_rgb_motion = old_last
    server._rgb_motion_previous_raw_frames = old_previous
    observation = _observation_with_raw(_raw_inputs())

    prepared = server._prepare_observed_rgb_motion(observation, frame_st_id=7)

    assert len(processor.calls) == 1
    assert server._last_rgb_motion is old_last
    assert server._rgb_motion_previous_raw_frames is old_previous
    assert prepared["sidecar"]["world_time_id"][0, 0, 0].item() == 7

    server._commit_prepared_rgb_motion(prepared)
    assert len(processor.calls) == 1
    assert server._last_rgb_motion is prepared["last_rgb_motion"]
    assert (
        server._rgb_motion_previous_raw_frames
        is prepared["previous_raw_frames"]
    )


def test_prediction_never_runs_online_raw_preprocessing():
    server = _server(online=True)
    processor = _RecordingRawPreprocessor()
    server._rgb_motion_preprocessor = processor
    server._rgb_motion_for_frames(
        _observation_with_raw(_raw_inputs(previous=True)),
        1,
        4,
        observed=True,
    )

    predicted = server._rgb_motion_for_frames(
        {"rgb_motion_inputs": _raw_inputs()},
        2,
        5,
        observed=False,
    )

    assert len(processor.calls) == 1
    assert predicted["world_time_id"][:, :, 0].tolist() == [[5, 6]]
    assert not predicted["observation_flag"].bool().any()


def test_online_require_previous_fails_clearly_on_cold_start():
    server = _server(online=True, first_frame_policy="require_previous")
    server._rgb_motion_preprocessor = _RecordingRawPreprocessor()

    with pytest.raises(ValueError, match="cold-start.*needs.*previous"):
        server._rgb_motion_for_frames(
            _observation_with_raw(_raw_inputs()),
            1,
            0,
            observed=True,
        )


def test_online_raw_inputs_require_camera_order_and_real_camera_geometry():
    server = _server(online=True)
    server._rgb_motion_preprocessor = _RecordingRawPreprocessor()

    missing_order = _raw_inputs(previous=True)
    missing_order.pop("camera_keys")
    with pytest.raises(KeyError, match="missing camera_keys"):
        server._rgb_motion_for_frames(
            {"rgb_motion_inputs": missing_order}, 1, 0, observed=True
        )

    missing_pose = _raw_inputs(previous=True)
    missing_pose["cameras"]["camera"].pop("world_from_camera")
    # A robot state is deliberately present and must never be accepted as a
    # camera pose fallback.
    with pytest.raises(KeyError, match="world_from_camera.*robot state"):
        server._rgb_motion_for_frames(
            {
                "rgb_motion_inputs": missing_pose,
                "state": torch.eye(4),
            },
            1,
            0,
            observed=True,
        )


def test_online_raw_streaming_anchors_are_derived_from_cache_not_world_time():
    server = _server(online=True, first_frame_policy="empty")
    processor = _RecordingRawPreprocessor()
    server._rgb_motion_preprocessor = processor

    # A cold encoder accepts exactly one seed and anchors its latent at raw 0,
    # even when the semantic world-time coordinate is nonzero.
    cold = _raw_inputs(include_anchor=False)
    prepared = server._prepare_observed_rgb_motion(
        _observation_with_raw(cold), frame_st_id=17
    )
    assert prepared["num_frames"] == 1
    assert processor.calls[-1][1]["anchor_indices"].tolist() == [0]
    assert processor.calls[-1][1]["world_time_ids"].tolist() == [17]

    # Conversely, a cached cold seed makes the first grounding encode warm
    # while frame_st_id can still be zero. T=8 at stride 4 yields causal ends
    # [3,7], not the offline whole-video schedule [0,4].
    server.streaming_vae.feat_cache = [torch.tensor([1.0])]
    warm = _raw_inputs(num_frames=8, include_anchor=False)
    prepared = server._prepare_observed_rgb_motion(
        _observation_with_raw(warm), frame_st_id=0
    )
    assert prepared["num_frames"] == 2
    assert processor.calls[-1][1]["anchor_indices"].tolist() == [3, 7]
    assert processor.calls[-1][1]["world_time_ids"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "warm,raw_frames,anchors,error",
    [
        (False, 2, None, "cold.*exactly one seed frame"),
        (True, 6, None, "warm.*divisible.*stride"),
        (True, 8, [0, 4], r"expected \[3, 7\].*got \[0, 4\]"),
    ],
)
def test_online_raw_rejects_noncausal_streaming_anchor_layout_before_dino(
    warm, raw_frames, anchors, error
):
    server = _server(online=True, first_frame_policy="empty")
    server.streaming_vae.feat_cache = [
        torch.tensor([1.0]) if warm else None
    ]

    class _MustNotRun:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("invalid streaming anchors reached DINO")

    server._rgb_motion_preprocessor = _MustNotRun()
    raw = _raw_inputs(num_frames=raw_frames, include_anchor=False)
    if anchors is not None:
        raw["anchor_indices"] = anchors
    with pytest.raises(ValueError, match=error):
        server._prepare_observed_rgb_motion(
            _observation_with_raw(raw), frame_st_id=0
        )


def test_online_raw_inputs_reject_integer_depth_and_pixel_misalignment():
    server = _server(online=True, first_frame_policy="empty")
    server._rgb_motion_preprocessor = _RecordingRawPreprocessor()

    integer_depth = _raw_inputs()
    integer_depth["cameras"]["camera"]["depth"] = torch.ones(
        1, 8, 8, dtype=torch.uint16
    )
    with pytest.raises(TypeError, match="floating-point z-depth"):
        server._rgb_motion_for_frames(
            {"rgb_motion_inputs": integer_depth}, 1, 0, observed=True
        )

    misaligned = _raw_inputs()
    misaligned["cameras"]["camera"]["depth"] = torch.ones(1, 7, 8)
    with pytest.raises(ValueError, match="RGB/depth spatial shapes must match"):
        server._rgb_motion_for_frames(
            {"rgb_motion_inputs": misaligned}, 1, 0, observed=True
        )


def test_online_explicit_previous_uses_the_same_geometry_validation():
    server = _server(online=True)
    server._rgb_motion_preprocessor = _RecordingRawPreprocessor()
    raw = _raw_inputs(previous=True)
    raw["previous"]["cameras"]["camera"]["depth"] = torch.ones(1, 6, 8)

    with pytest.raises(ValueError, match="previous.*RGB/depth spatial shapes"):
        server._rgb_motion_for_frames(
            _observation_with_raw(raw), 1, 0, observed=True
        )


def test_online_world_time_must_match_server_wan_step_coordinates():
    server = _server(online=True)
    server._rgb_motion_preprocessor = _RecordingRawPreprocessor()
    raw = _raw_inputs(previous=True)
    raw["world_time_ids"] = [99]

    with pytest.raises(ValueError, match="must equal the server's grounded WAN-step"):
        server._rgb_motion_for_frames(
            _observation_with_raw(raw), 1, 7, observed=True
        )


def test_server_rejects_fractional_indices_and_nonbinary_masks():
    server = _server()
    fractional = _one_frame_payload()
    fractional["motion_indices"] = fractional["motion_indices"].float()
    fractional["motion_indices"][0, 0] = 1.5
    with pytest.raises(ValueError, match="finite integer values"):
        server._rgb_motion_for_frames(
            {"rgb_motion": fractional}, 1, 0, observed=True
        )

    nonbinary = _one_frame_payload()
    nonbinary["motion_valid_mask"] = torch.tensor([[1, 2, 1]])
    with pytest.raises(ValueError, match="only 0 or 1"):
        server._rgb_motion_for_frames(
            {"rgb_motion": nonbinary}, 1, 0, observed=True
        )


@pytest.mark.parametrize(
    "field, mutate, error",
    [
        (
            "motion_scores",
            lambda payload: payload["motion_scores"].__setitem__(
                (0, 0), float("nan")
            ),
            "motion_scores must be finite",
        ),
        (
            "dino_features",
            lambda payload: payload["dino_features"].__setitem__(
                (0, 0, 0), float("inf")
            ),
            "dino_features must be finite",
        ),
    ],
)
def test_server_rejects_nonfinite_valid_metadata(field, mutate, error):
    del field
    server = _server()
    payload = _one_frame_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=error):
        server._rgb_motion_for_frames(
            {"rgb_motion": payload}, 1, 0, observed=True
        )


def test_presence_masks_require_nonempty_finite_feature_channels():
    server = _server()
    no_visual_width = _one_frame_payload()
    no_visual_width["dino_features"] = torch.empty(1, 3, 0)
    with pytest.raises(ValueError, match="non-zero width.*visual_valid"):
        server._rgb_motion_for_frames(
            {"rgb_motion": no_visual_width}, 1, 0, observed=True
        )

    no_tactile_width = _one_frame_payload()
    no_tactile_width["tactile_valid"] = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="non-zero width.*tactile_valid"):
        server._rgb_motion_for_frames(
            {"rgb_motion": no_tactile_width}, 1, 0, observed=True
        )

    bad_tactile = _one_frame_payload()
    bad_tactile["neoforce_features"] = torch.tensor(
        [[[float("nan")], [0.0], [1.0]]]
    )
    bad_tactile["tactile_valid"] = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="neoforce_features must be finite"):
        server._rgb_motion_for_frames(
            {"rgb_motion": bad_tactile}, 1, 0, observed=True
        )

    missing_tactile_mask = _one_frame_payload()
    missing_tactile_mask["neoforce_features"] = torch.ones(1, 3, 2)
    missing_tactile_mask.pop("tactile_valid")
    with pytest.raises(KeyError, match="missing tactile_valid"):
        server._rgb_motion_for_frames(
            {"rgb_motion": missing_tactile_mask}, 1, 0, observed=True
        )


def test_server_rejects_duplicate_valid_motion_addresses():
    server = _server()
    payload = _one_frame_payload()
    payload["motion_indices"] = torch.tensor([[1, -1, 1]])
    with pytest.raises(ValueError, match="duplicate valid addresses"):
        server._rgb_motion_for_frames(
            {"rgb_motion": payload}, 1, 0, observed=True
        )


def test_observed_sidecar_has_independent_canonical_time_and_source():
    server = _server()
    sidecar = server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 12, observed=True
    )

    assert sidecar["motion_indices"].shape == (1, 1, 3)
    assert sidecar["dino_features"].shape == (1, 1, 3, 2)
    assert sidecar["neoforce_features"].shape == (1, 1, 3, 0)
    assert sidecar["world_time_id"].tolist() == [[[12, -1, 12]]]
    assert sidecar["observation_flag"].tolist() == [[[1, 0, 1]]]
    assert server._last_rgb_motion["world_time_id"].tolist() == [[12, -1, 12]]
    # The sidecar remains separate metadata; there is no content-embedding field.
    assert "noisy_latents" not in sidecar


def test_observed_sidecar_requires_exact_grounding_frame_count():
    server = _server()
    with pytest.raises(ValueError, match="exactly one row per grounded WAN frame"):
        server._rgb_motion_for_frames(
            {"rgb_motion": _one_frame_payload()}, 2, 12, observed=True
        )

    exact = server._rgb_motion_for_frames(
        {"rgb_motion": _payload_for_frames(2)}, 2, 12, observed=True
    )
    assert exact["motion_indices"].shape == (1, 2, 3)


def test_prediction_reuses_latest_support_and_extends_every_field():
    server = _server()
    observed = server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 4, observed=True
    )
    predicted = server._rgb_motion_for_frames({}, 3, 5, observed=False)

    assert predicted["motion_indices"].shape == (1, 3, 3)
    assert predicted["dino_features"].shape == (1, 3, 3, 2)
    assert predicted["neoforce_features"].shape == (1, 3, 3, 0)
    for frame in range(3):
        torch.testing.assert_close(
            predicted["motion_indices"][0, frame],
            observed["motion_indices"][0, 0],
        )
        torch.testing.assert_close(
            predicted["dino_features"][0, frame],
            observed["dino_features"][0, 0],
        )
    assert predicted["world_time_id"][:, :, 0].tolist() == [[5, 6, 7]]
    assert not predicted["observation_flag"].bool().any()


def test_cold_seed_overlay_changes_only_leading_frame_not_k_tokens():
    server = _server()
    observed_seed = server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 0, observed=True
    )
    predicted = server._rgb_motion_for_frames({}, 3, 0, observed=False)

    merged = server._overlay_rgb_motion_seed_frames(predicted, observed_seed)

    assert merged["motion_indices"].shape == (1, 3, 3)  # F=3, K=3
    assert merged["observation_flag"].bool().any(dim=-1).tolist() == [
        [True, False, False]
    ]
    assert merged["world_time_id"][0, :, 0].tolist() == [0, 1, 2]


def test_prediction_fallback_repeats_only_newest_observed_support():
    server = _server()
    payload = _one_frame_payload()
    payload = {
        name: torch.cat([value, value.clone()], dim=0)
        for name, value in payload.items()
    }
    payload["motion_indices"][0] = torch.tensor([0, -1, 2])
    payload["motion_indices"][1] = torch.tensor([1, -1, 3])
    payload["dino_features"][0, 0] = torch.tensor([2.0, 0.0])
    payload["dino_features"][1, 0] = torch.tensor([0.0, 2.0])

    observed = server._rgb_motion_for_frames(
        {"rgb_motion": payload}, 2, 4, observed=True
    )
    predicted = server._rgb_motion_for_frames({}, 2, 6, observed=False)

    # A cached observation is a held support state, not a trajectory to replay.
    for frame in range(2):
        torch.testing.assert_close(
            predicted["motion_indices"][0, frame],
            observed["motion_indices"][0, -1],
        )
        torch.testing.assert_close(
            predicted["dino_features"][0, frame],
            observed["dino_features"][0, -1],
        )


def test_explicit_future_payload_preserves_its_per_frame_support():
    server = _server()
    payload = _one_frame_payload()
    payload = {
        name: torch.cat([value, value.clone()], dim=0)
        for name, value in payload.items()
    }
    payload["motion_indices"][0] = torch.tensor([0, -1, 2])
    payload["motion_indices"][1] = torch.tensor([1, -1, 3])

    predicted = server._rgb_motion_for_frames(
        {"rgb_motion": payload}, 2, 8, observed=False
    )

    assert predicted["motion_indices"][0].tolist() == [
        [0, -1, 2],
        [1, -1, 3],
    ]


def test_invalid_positive_indices_are_canonicalized_to_minus_one():
    server = _server()
    payload = _one_frame_payload()
    payload["motion_indices"][0, 1] = 10_000
    payload["motion_valid_mask"][0, 1] = False

    sidecar = server._rgb_motion_for_frames(
        {"rgb_motion": payload}, 1, 3, observed=True
    )

    assert sidecar["motion_indices"][0, 0, 1].item() == -1


def test_prediction_can_extend_payload_when_optional_scores_are_absent():
    server = _server()
    payload = _one_frame_payload()
    payload.pop("motion_scores")

    predicted = server._rgb_motion_for_frames(
        {"rgb_motion": payload}, 3, 5, observed=False
    )

    assert predicted["motion_scores"].shape == (1, 3, 3)
    assert predicted["motion_scores"].tolist() == [
        [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    ]


def test_cfg_repeats_content_and_every_index_component_in_lockstep():
    server = _server()
    server.use_cfg = True
    server.prompt_embeds = torch.tensor([[[1.0]]])
    server.negative_prompt_embeds = torch.tensor([[[-1.0]]])
    sidecar = server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 2, observed=False
    )
    input_dict = {
        "noisy_latents": torch.zeros(1, 2, 1, 2, 2),
        "text_emb": torch.zeros(1, 1, 1),
        "grid_id": torch.zeros(4, 1),
        "timesteps": torch.zeros(1),
        **sidecar,
    }

    result = server._repeat_input_for_cfg(input_dict)

    assert result["noisy_latents"].shape[0] == 2
    assert result["text_emb"][:, 0, 0].tolist() == [1.0, -1.0]
    for key in (
        "motion_indices",
        "motion_valid_mask",
        "motion_scores",
        "world_time_id",
        "dino_features",
        "neoforce_features",
        "observation_flag",
        "visual_valid",
        "tactile_valid",
    ):
        assert result[key].shape[0] == 2
        torch.testing.assert_close(result[key][0], result[key][1])


def test_sparse_projection_scatter_uses_local_frame_indices_and_wan_order():
    server = _server()
    # Grid is (F=2,H=1,W=2), so local index 1 at frame 0 is global patch 1,
    # and local index 0 at frame 1 is global patch 2.
    template = torch.zeros(1, 2, 2, 2, 4)
    motion = {
        "motion_indices": torch.tensor([[[1, -1], [0, -1]]]),
        "motion_valid_mask": torch.tensor(
            [[[True, False], [True, False]]]
        ),
    }
    patch_volume = 4
    prediction = torch.arange(
        1 * 4 * patch_volume * 2, dtype=torch.float32
    ).reshape(1, 4 * patch_volume, 2)

    dense = server._sparse_video_prediction_to_dense(
        prediction, template, motion
    )
    dense_patches, grid = patchify_latents(dense, (1, 2, 2))
    raw = prediction.reshape(1, 4, patch_volume, 2).permute(
        0, 1, 3, 2
    ).reshape(1, 4, -1)

    assert grid == (2, 1, 2)
    torch.testing.assert_close(dense_patches[0, 1], raw[0, 0])
    torch.testing.assert_close(dense_patches[0, 2], raw[0, 2])
    torch.testing.assert_close(dense_patches[0, 0], torch.zeros(8))
    torch.testing.assert_close(dense_patches[0, 3], torch.zeros(8))


def test_sparse_projection_scatter_supports_cfg_batch():
    server = _server()
    template = torch.zeros(2, 1, 1, 2, 2)
    motion = {
        "motion_indices": torch.tensor([[[0]], [[0]]]),
        "motion_valid_mask": torch.ones(2, 1, 1, dtype=torch.bool),
    }
    prediction = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[10.0], [20.0], [30.0], [40.0]],
        ]
    )

    dense = server._sparse_video_prediction_to_dense(
        prediction, template, motion
    )

    assert dense.shape == template.shape
    torch.testing.assert_close(dense[0].flatten(), prediction[0].flatten())
    torch.testing.assert_close(dense[1].flatten(), prediction[1].flatten())


def test_sparse_canvas_keeps_observed_background_outside_selected_patches():
    server = _server()
    # Deliberately provide two observed frames: only the newest one is the
    # static-world background broadcast over the future chunk.
    server.init_latent = None
    server._last_observed_video_latent = torch.stack(
        [torch.full((1, 2, 4), 3.0), torch.full((1, 2, 4), 7.0)], dim=1
    ).unsqueeze(0)
    noise = torch.arange(1 * 1 * 2 * 2 * 4, dtype=torch.float32).reshape(
        1, 1, 2, 2, 4
    )
    motion = {
        "motion_indices": torch.tensor([[[1], [0]]]),
        "motion_valid_mask": torch.ones(1, 2, 1, dtype=torch.bool),
    }

    background = server._rgb_motion_background(noise)
    canvas = server._sparse_canvas_from_dense(noise, background, motion)
    canvas_patches, _ = patchify_latents(canvas, (1, 2, 2))
    noise_patches, _ = patchify_latents(noise, (1, 2, 2))
    background_patches, _ = patchify_latents(background, (1, 2, 2))

    torch.testing.assert_close(background, torch.full_like(background, 7.0))
    # Selected local (frame, spatial) cells: (0,1) -> global 1,
    # (1,0) -> global 2.  Everything else must remain the real background.
    torch.testing.assert_close(canvas_patches[0, 1], noise_patches[0, 1])
    torch.testing.assert_close(canvas_patches[0, 2], noise_patches[0, 2])
    torch.testing.assert_close(canvas_patches[0, 0], background_patches[0, 0])
    torch.testing.assert_close(canvas_patches[0, 3], background_patches[0, 3])


def test_prepare_latent_input_forwards_sidecar_only_to_video_branch():
    server = _server()
    server.prompt_embeds = torch.zeros(1, 2, 3)
    server.action_mask = torch.ones(2, dtype=torch.bool)
    latent = torch.zeros(1, 1, 1, 2, 2)
    action = torch.zeros(1, 2, 1, 1, 1)
    sidecar = server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 9, observed=True
    )

    prepared = server._prepare_latent_input(
        latent,
        action,
        frame_st_id=9,
        rgb_motion=sidecar,
    )

    for key, value in sidecar.items():
        assert prepared["latent_res_lst"][key] is value
        assert key not in prepared["action_res_lst"]


def _consistency_server(
    tmp_path,
    *,
    enabled,
    live_cameras,
    metadata,
    patch_size=(1, 2, 2),
):
    bundle = tmp_path / "bundle"
    (bundle / "transformer").mkdir(parents=True)
    (bundle / "train_meta.json").write_text(json.dumps(metadata))
    server = TWAMServer.__new__(TWAMServer)
    server.job_config = SimpleNamespace(
        wan22_pretrained_model_name_or_path=str(bundle),
        norm_stat={},
        use_rgb_motion_tokens=enabled,
        obs_cam_keys=live_cameras,
        patch_size=patch_size,
    )
    return server


def test_rgb_motion_consistency_requires_exact_camera_order(tmp_path):
    metadata = {
        "use_rgb_motion_tokens": True,
        "obs_cam_keys": ["left", "right"],
    }
    matching = _consistency_server(
        tmp_path / "matching",
        enabled=True,
        live_cameras=["left", "right"],
        metadata=metadata,
    )
    matching._check_train_serve_consistency()

    reversed_order = _consistency_server(
        tmp_path / "reversed",
        enabled=True,
        live_cameras=["right", "left"],
        metadata=metadata,
    )
    try:
        reversed_order._check_train_serve_consistency()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("RGB-motion camera-order mismatch was not rejected")
    assert "obs_cam_keys order differs" in message

    unverifiable = _consistency_server(
        tmp_path / "missing",
        enabled=True,
        live_cameras=["left", "right"],
        metadata={"use_rgb_motion_tokens": True},
    )
    try:
        unverifiable._check_train_serve_consistency()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing RGB-motion camera metadata was not rejected")
    assert "camera-grid order cannot be verified" in message


def test_dense_consistency_ignores_camera_order_and_old_missing_field(tmp_path):
    mismatched = _consistency_server(
        tmp_path / "mismatched",
        enabled=False,
        live_cameras=["right", "left"],
        metadata={"obs_cam_keys": ["left", "right"]},
    )
    mismatched._check_train_serve_consistency()

    old_metadata = _consistency_server(
        tmp_path / "old",
        enabled=False,
        live_cameras=["camera"],
        metadata={},
    )
    old_metadata._check_train_serve_consistency()


def test_consistency_compares_patch_size_when_new_metadata_has_it(tmp_path):
    metadata = {"patch_size": [1, 2, 2]}
    matching = _consistency_server(
        tmp_path / "matching_patch",
        enabled=False,
        live_cameras=["camera"],
        metadata=metadata,
        patch_size=(1, 2, 2),
    )
    matching._check_train_serve_consistency()

    mismatched = _consistency_server(
        tmp_path / "mismatched_patch",
        enabled=False,
        live_cameras=["camera"],
        metadata=metadata,
        patch_size=(1, 4, 2),
    )
    try:
        mismatched._check_train_serve_consistency()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("patch_size mismatch was not rejected")
    assert "patch_size" in message


def test_reset_uses_sparse_token_budget_and_drops_prior_episode_state(tmp_path):
    class _Transformer:
        def __init__(self):
            self.created = None
            self.cleared = None

        def clear_cache(self, name):
            self.cleared = name

        def create_empty_cache(self, *args, **kwargs):
            self.created = (args, kwargs)

    class _StreamingVAE:
        def __init__(self):
            self.cleared = False

        def clear_cache(self):
            self.cleared = True

    server = _server(patch_size=(1, 2, 2))
    server.cache_name = "pos"
    server.transformer = _Transformer()
    server.streaming_vae = _StreamingVAE()
    server._reset_tactile_state = lambda: None
    server._last_rgb_motion = {"stale": torch.tensor(1)}
    server._last_observed_video_latent = torch.ones(1)
    server._rgb_motion_previous_raw_frames = {"stale": torch.tensor(1)}
    server.save_root = str(tmp_path)
    server.job_config = SimpleNamespace(
        use_rgb_motion_tokens=True,
        rgb_motion_max_tokens=3,
        patch_size=(1, 2, 2),
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        frame_chunk_size=6,
        action_per_frame=4,
        height=32,
        width=32,
        obs_cam_keys=["left", "right"],
        tactile_keys=[],
        attn_window=4,
        action_dim=5,
        used_action_channel_ids=[0, 2],
        norm_stat={"q01": [0.0] * 5, "q99": [1.0] * 5},
        action_norm_method="quantiles",
        prompt=None,
    )

    server._reset()

    assert server.frame_st_id == 0
    assert server._last_rgb_motion is None
    assert server._last_observed_video_latent is None
    assert server._rgb_motion_previous_raw_frames is None
    assert server.transformer.cleared == "pos"
    assert server.streaming_vae.cleared
    args, kwargs = server.transformer.created
    assert args[:2] == ("pos", 4)
    # 6 temporal latent frames * K=3 (RGB serving requires p_t=1).
    assert args[2] == 18
    assert args[3] == 6 * 4
    assert kwargs["batch_size"] == 1


def test_rgb_motion_server_rejects_temporal_patchification():
    server = _server(patch_size=(2, 2, 2))
    server.job_config = SimpleNamespace(
        use_rgb_motion_tokens=True,
        patch_size=(2, 2, 2),
    )
    with pytest.raises(ValueError, match="temporal patch_size=1"):
        server._validate_rgb_motion_server_config(server.job_config)


def test_online_rgb_motion_requires_sparse_mode_and_positive_fixed_budget():
    with pytest.raises(ValueError, match="requires use_rgb_motion_tokens"):
        TWAMServer._validate_rgb_motion_server_config(SimpleNamespace(
            use_rgb_motion_tokens=False,
            rgb_motion_online_preprocess=True,
        ))

    with pytest.raises(ValueError, match="rgb_motion_max_tokens > 0"):
        TWAMServer._validate_rgb_motion_server_config(SimpleNamespace(
            use_rgb_motion_tokens=True,
            rgb_motion_online_preprocess=True,
            rgb_motion_max_tokens=0,
            rgb_motion_first_frame_policy="empty",
            obs_cam_keys=["camera"],
            patch_size=(1, 2, 2),
        ))


def test_online_preprocessor_lazy_loads_dino_local_only(monkeypatch):
    from n0_twam.preprocessing.dinov2 import FrozenDinoV2PatchEncoder

    calls = []

    class _FakeDino:
        def __call__(self, rgb):
            return torch.zeros(rgb.shape[0], 2, 2, 4)

    def fake_from_pretrained(source, **kwargs):
        calls.append((source, kwargs))
        return _FakeDino()

    monkeypatch.setattr(
        FrozenDinoV2PatchEncoder, "from_pretrained", fake_from_pretrained
    )
    server = _server(online=True)
    server.job_config.rgb_motion_dino_model_name_or_path = "/local/dinov2"

    first = server._get_rgb_motion_preprocessor()
    second = server._get_rgb_motion_preprocessor()

    assert first is second
    assert len(calls) == 1
    assert calls[0][0] == "/local/dinov2"
    assert calls[0][1]["local_files_only"] is True
    assert first.detector.max_tokens is None
    assert first.camera_keys == ("camera",)


def test_prepare_latent_input_uses_configured_patch_size():
    server = _server(patch_size=(1, 4, 2))
    server.prompt_embeds = torch.zeros(1, 1, 1)
    calls = []
    method_globals = TWAMServer._prepare_latent_input.__globals__
    original_get_mesh_id = method_globals["get_mesh_id"]

    def recording_mesh_id(frames, height, width, *args, **kwargs):
        calls.append((frames, height, width))
        return torch.zeros(4, frames * height * width)

    method_globals["get_mesh_id"] = recording_mesh_id
    try:
        prepared = server._prepare_latent_input(
            torch.zeros(1, 2, 3, 8, 10),
            None,
        )
    finally:
        method_globals["get_mesh_id"] = original_get_mesh_id

    assert calls == [(3, 2, 5)]
    assert prepared["latent_res_lst"]["grid_id"].shape == (4, 30)


def test_reset_initializes_negative_prompt_for_action_only_cfg(tmp_path):
    class _Transformer:
        def clear_cache(self, _name):
            pass

        def create_empty_cache(self, *args, **kwargs):
            self.batch_size = kwargs["batch_size"]

    class _StreamingVAE:
        def clear_cache(self):
            pass

    server = _server(enabled=False)
    server.cache_name = "pos"
    server.transformer = _Transformer()
    server.streaming_vae = _StreamingVAE()
    server._reset_tactile_state = lambda: None
    server.save_root = str(tmp_path)
    server.job_config = SimpleNamespace(
        use_rgb_motion_tokens=False,
        patch_size=(1, 2, 2),
        guidance_scale=1.0,
        action_guidance_scale=2.0,
        frame_chunk_size=2,
        action_per_frame=1,
        height=32,
        width=32,
        obs_cam_keys=["camera"],
        tactile_keys=[],
        attn_window=4,
        action_dim=2,
        used_action_channel_ids=[0, 1],
        norm_stat={"q01": [0.0, 0.0], "q99": [1.0, 1.0]},
        action_norm_method="quantiles",
        prompt="task",
    )
    captured = {}

    def _encode_prompt(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[[1.0]]]), torch.tensor([[[-1.0]]])

    server.encode_prompt = _encode_prompt
    server._reset(prompt="task")

    assert server.use_cfg
    assert captured["do_classifier_free_guidance"] is True
    assert server.transformer.batch_size == 2
    repeated = server._repeat_input_for_cfg({
        "noisy_latents": torch.zeros(1, 1, 1, 2, 2),
        "text_emb": torch.zeros(1, 1, 1),
        "grid_id": torch.zeros(4, 1),
        "timesteps": torch.zeros(1),
    })
    assert repeated["text_emb"][:, 0, 0].tolist() == [1.0, -1.0]


class _CacheTransformer:
    def __init__(self):
        self.clear_calls = []
        self.forward_calls = []

    def clear_pred_cache(self, name):
        self.clear_calls.append(name)

    def __call__(self, input_dict, **kwargs):
        self.forward_calls.append((input_dict, kwargs))
        return torch.empty(0)


class _FailingActionCacheTransformer(_CacheTransformer):
    """Tiny transactional cache double for the server's two-pass grounding."""

    def __init__(self):
        super().__init__()
        self.committed = ["old-prediction"]
        self.transaction_events = []

    def clear_pred_cache(self, name):
        super().clear_pred_cache(name)
        self.committed.clear()

    def cache_transaction(self, name):
        owner = self

        class _Transaction:
            def __enter__(self):
                self.snapshot = list(owner.committed)
                owner.transaction_events.append(("begin", name))
                return self

            def __exit__(self, exc_type, _exc, _traceback):
                if exc_type is None:
                    owner.transaction_events.append(("commit", name))
                else:
                    owner.committed[:] = self.snapshot
                    owner.transaction_events.append(("rollback", name))
                return False

        return _Transaction()

    def __call__(self, input_dict, **kwargs):
        super().__call__(input_dict, **kwargs)
        branch = input_dict["branch"]
        self.committed.append(branch)
        if kwargs["action_mode"]:
            raise RuntimeError("forced action grounding failure")
        return torch.empty(0)


def _kv_lifecycle_server(*, sparse: bool):
    server = _server(enabled=sparse)
    server.job_config.rgb_motion_max_tokens = 3
    server.cache_name = "pos"
    server.frame_st_id = 0
    server.exp_save_root = "/tmp"
    server.init_latent = torch.full((1, 1, 1, 2, 2), -5.0)
    server.transformer = _CacheTransformer()
    server._last_gen_tactile = None
    server._last_gen_tactile_fsid = None
    server._encode_obs = lambda obs: torch.stack(
        [torch.full((1, 2, 2), 10.0), torch.full((1, 2, 2), 20.0)], dim=1
    ).unsqueeze(0)
    server.preprocess_action = lambda *args, **kwargs: torch.arange(
        3, dtype=torch.float32
    ).reshape(1, 1, 3, 1, 1).expand(1, 2, -1, -1, -1).clone()
    server._encode_tactile_obs = lambda obs: None
    server._repeat_input_for_cfg = lambda value: value
    captured = {}

    def _prepare(latent, action, **kwargs):
        captured.update(latent=latent, action=action, kwargs=kwargs)
        return {"latent_res_lst": {"branch": "video"},
                "action_res_lst": {"branch": "action"}}

    server._prepare_latent_input = _prepare
    return server, captured


def test_first_grounding_after_cold_seed_uses_warm_streaming_anchors():
    server, captured = _kv_lifecycle_server(sparse=True)
    server.job_config.rgb_motion_online_preprocess = True
    server.job_config.rgb_motion_first_frame_policy = "empty"
    processor = _RecordingRawPreprocessor()
    server._rgb_motion_preprocessor = processor

    # Simulate the successful cold imagination: semantic/VAE seed state is
    # cached, but grounding still enters with frame_st_id == 0.
    server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 0, observed=True
    )
    server.streaming_vae = _TestStreamingVAE(warm=True)
    server.vae = server.streaming_vae.vae

    raw = _raw_inputs(num_frames=8, include_anchor=False)
    observation = _observation_with_raw(raw)
    observation["state"] = torch.zeros(2, 3, 1)
    server._compute_kv_cache(observation)

    assert processor.calls[0][1]["anchor_indices"].tolist() == [3, 7]
    assert processor.calls[0][1]["world_time_ids"].tolist() == [1, 2]
    assert captured["kwargs"]["frame_st_id"] == 1
    assert captured["latent"].shape[2] == 2
    assert server.frame_st_id == 3


def test_sparse_cold_grounding_keeps_cached_seed_without_appending_it_twice():
    server, captured = _kv_lifecycle_server(sparse=True)
    # Model the successful cold-generation transaction: the clean seed has an
    # observed semantic row in cache and its canonical sidecar remains available.
    server._rgb_motion_for_frames(
        {"rgb_motion": _one_frame_payload()}, 1, 0, observed=True
    )

    server._compute_kv_cache(
        {
            "obs": [],
            "state": torch.zeros(2, 3, 1),
            "rgb_motion": _payload_for_frames(2),
        }
    )

    # Only the two new real frames are written.  The already cached seed at t=0
    # is not prefixed a second time.
    assert captured["latent"].shape[2] == 2
    assert captured["latent"][:, :, 0].eq(10).all()
    assert captured["kwargs"]["frame_st_id"] == 1
    index = captured["kwargs"]["rgb_motion"]
    assert index["world_time_id"][:, :, 0].tolist() == [[1, 2]]
    assert index["observation_flag"][index["motion_valid_mask"]].bool().all()
    assert not index["observation_flag"][~index["motion_valid_mask"]].bool().any()
    # The client's full action chunk still contains the unexecuted cold frame0;
    # it must be removed so executed actions 1.. align with video times 1.. .
    assert captured["action"].shape[2] == 2
    assert captured["action"][0, 0, :, 0, 0].tolist() == [1.0, 2.0]
    assert server.frame_st_id == 3
    assert server.transformer.clear_calls == ["pos"]
    assert len(server.transformer.forward_calls) == 2


def test_dense_legacy_cold_grounding_still_reappends_seed_after_pred_clear():
    server, captured = _kv_lifecycle_server(sparse=False)

    server._compute_kv_cache({"obs": [], "state": torch.zeros(2, 3, 1)})

    assert captured["latent"].shape[2] == 3
    assert captured["latent"][:, :, 0].eq(-5).all()
    assert captured["latent"][:, :, 1].eq(10).all()
    assert captured["latent"][:, :, 2].eq(20).all()
    assert captured["action"].shape[2] == 3
    assert captured["action"][0, 0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert captured["kwargs"]["frame_st_id"] == 0
    assert captured["kwargs"]["rgb_motion"] is None
    assert server.frame_st_id == 3


def test_grounding_video_and_action_cache_updates_are_one_transaction():
    server, _captured = _kv_lifecycle_server(sparse=False)
    transformer = _FailingActionCacheTransformer()
    server.transformer = transformer

    old_video_cache = torch.tensor([1.0])
    old_global_cache = torch.tensor([2.0])
    old_local_cache = torch.tensor([3.0])
    server.streaming_vae = SimpleNamespace(feat_cache=[old_video_cache])
    server.tactile_global_vae = SimpleNamespace(feat_cache=[old_global_cache])
    server.tactile_local_vae = SimpleNamespace(feat_cache=[old_local_cache])
    old_first = torch.tensor([4.0])
    old_previous = torch.tensor([5.0])
    old_tactile_latents = {"old": torch.tensor([6.0])}
    old_observed_latent = torch.tensor([7.0])
    server.tactile_first_frames = old_first
    server.tactile_prev_frames = old_previous
    server.last_tactile_latents = old_tactile_latents
    server._last_observed_video_latent = old_observed_latent

    def _encode_obs(_obs):
        server.streaming_vae.feat_cache[0] = torch.tensor([10.0])
        return torch.stack(
            [
                torch.full((1, 2, 2), 10.0),
                torch.full((1, 2, 2), 20.0),
            ],
            dim=1,
        ).unsqueeze(0)

    def _encode_tactile(_obs):
        server.tactile_global_vae.feat_cache[0] = torch.tensor([20.0])
        server.tactile_local_vae.feat_cache[0] = torch.tensor([30.0])
        server.tactile_first_frames = torch.tensor([40.0])
        server.tactile_prev_frames = torch.tensor([50.0])
        return {"candidate": torch.tensor([60.0])}

    server._encode_obs = _encode_obs
    server._encode_tactile_obs = _encode_tactile

    with pytest.raises(RuntimeError, match="forced action grounding failure"):
        server._compute_kv_cache(
            {"obs": [], "state": torch.zeros(2, 3, 1)}
        )

    # The video call succeeded before action failed, but the request-level
    # transaction restores the common cache rather than exposing video-only KV.
    # The request transaction began before prediction clearing, so the old
    # prediction is restored together with the successful video append.
    assert transformer.committed == ["old-prediction"]
    assert transformer.transaction_events == [
        ("begin", "pos"),
        ("rollback", "pos"),
    ]
    assert [call[1]["action_mode"] for call in transformer.forward_calls] == [
        False,
        True,
    ]
    assert server.frame_st_id == 0
    # Streaming WAN/tactile VAE entries are replaced, never mutated in place;
    # shallow container snapshots restore the exact pre-request tensors without
    # a second GPU-sized tensor copy.
    assert server.streaming_vae.feat_cache[0] is old_video_cache
    assert server.tactile_global_vae.feat_cache[0] is old_global_cache
    assert server.tactile_local_vae.feat_cache[0] is old_local_cache
    assert server.tactile_first_frames is old_first
    assert server.tactile_prev_frames is old_previous
    assert server.last_tactile_latents is old_tactile_latents
    assert server._last_observed_video_latent is old_observed_latent


def test_plain_infer_postprocess_failure_rolls_back_cache_and_cold_state(
    monkeypatch,
):
    """A late output error must not publish either expert's predicted KV."""

    class _StreamingCache:
        def __init__(self, value):
            self.feat_cache = [value]

        def clear_cache(self):
            self.feat_cache = []

    class _Scheduler:
        def set_timesteps(self, _steps):
            self.timesteps = torch.tensor([0])

        def step(self, _prediction, _timestep, sample, return_dict=False):
            assert return_dict is False
            return sample

    class _InferenceTransformer(_FailingActionCacheTransformer):
        def __call__(self, input_dict, **kwargs):
            _CacheTransformer.__call__(self, input_dict, **kwargs)
            if kwargs["update_cache"]:
                self.committed.append(input_dict["branch"])
            if kwargs["action_mode"]:
                return torch.zeros(1, 1, 1)
            return torch.zeros(1, 1, 1)

    server = _server(enabled=False)
    server.cache_name = "pos"
    server.frame_st_id = 0
    server.exp_save_root = "/tmp"
    server.latent_height = 2
    server.latent_width = 2
    server.action_per_frame = 1
    server.use_cfg = False
    server.action_mask = torch.tensor([True])
    server.job_config.frame_chunk_size = 1
    server.job_config.tactile_keys = []
    server.job_config.server_tactile_denoise = False
    server.job_config.action_dim = 1
    server.job_config.action_delta_mode = "pi05_delta"
    server.job_config.num_inference_steps = 1
    server.job_config.action_num_inference_steps = 1
    server.job_config.video_exec_step = -1
    server.job_config.guidance_scale = 1.0
    server.job_config.action_guidance_scale = 1.0
    server.scheduler = _Scheduler()
    server.action_scheduler = _Scheduler()
    server.transformer = _InferenceTransformer()

    old_video_cache = torch.tensor([1.0])
    old_global_cache = torch.tensor([2.0])
    old_local_cache = torch.tensor([3.0])
    server.streaming_vae = _StreamingCache(old_video_cache)
    server.tactile_global_vae = _StreamingCache(old_global_cache)
    server.tactile_local_vae = _StreamingCache(old_local_cache)
    old_first = torch.tensor([4.0])
    old_previous = torch.tensor([5.0])
    old_tactile_latents = {"old": torch.tensor([6.0])}
    old_init = torch.tensor([7.0])
    old_observed_latent = torch.tensor([8.0])
    old_generated_tactile = torch.tensor([9.0])
    old_delta_smooth = torch.tensor([9.5])
    old_rgb = {"old": torch.tensor([10.0])}
    old_raw = {"old": torch.tensor([11.0])}
    server.tactile_first_frames = old_first
    server.tactile_prev_frames = old_previous
    server.last_tactile_latents = old_tactile_latents
    server.init_latent = old_init
    server._last_observed_video_latent = old_observed_latent
    server._last_gen_tactile = old_generated_tactile
    server._last_gen_tactile_fsid = 17
    server._delta_smooth_prev = old_delta_smooth
    server._last_rgb_motion = old_rgb
    server._rgb_motion_previous_raw_frames = old_raw

    def _encode_obs(_obs):
        server.streaming_vae.feat_cache = [torch.tensor([20.0])]
        return torch.zeros(1, 1, 1, 2, 2)

    server._encode_obs = _encode_obs
    server._prepare_latent_input = lambda latent, action, *args, **kwargs: (
        {"latent_res_lst": {"branch": "video"}}
        if latent is not None
        else {"action_res_lst": {"branch": "action"}}
    )
    server._repeat_input_for_cfg = lambda value: value
    monkeypatch.setitem(
        TWAMServer._infer_impl.__globals__,
        "data_seq_to_patch",
        lambda *args, **kwargs: torch.zeros(1, 48, 1, 2, 2),
    )

    def _fail_postprocess(*args, **kwargs):
        # The real pi0.5 postprocessor writes this field before it validates
        # the requested output format, so it belongs to the same undo record.
        server._delta_smooth_prev = torch.tensor([-1.0])
        raise RuntimeError("forced postprocess failure")

    server.postprocess_action = _fail_postprocess

    with pytest.raises(RuntimeError, match="forced postprocess failure"):
        server._infer({}, frame_st_id=0)

    # Both update_cache=1 calls ran, but postprocessing failed before the
    # request transaction committed, so the pre-request cache is restored.
    committed_calls = [
        call for call in server.transformer.forward_calls
        if call[1]["update_cache"] == 1
    ]
    assert [call[1]["action_mode"] for call in committed_calls] == [False, True]
    assert server.transformer.committed == ["old-prediction"]
    assert server.transformer.transaction_events == [
        ("begin", "pos"),
        ("rollback", "pos"),
    ]
    assert server.streaming_vae.feat_cache[0] is old_video_cache
    assert server.tactile_global_vae.feat_cache[0] is old_global_cache
    assert server.tactile_local_vae.feat_cache[0] is old_local_cache
    assert server.tactile_first_frames is old_first
    assert server.tactile_prev_frames is old_previous
    assert server.last_tactile_latents is old_tactile_latents
    assert server.init_latent is old_init
    assert server._last_observed_video_latent is old_observed_latent
    assert server._last_gen_tactile is old_generated_tactile
    assert server._last_gen_tactile_fsid == 17
    assert server._delta_smooth_prev is old_delta_smooth
    assert server._last_rgb_motion is old_rgb
    assert server._rgb_motion_previous_raw_frames is old_raw
    assert server.frame_st_id == 0
