from types import SimpleNamespace

import pytest
import torch

from n0_twam.models.rgb_motion import EgoMotionCompensatedMotionDetector
from n0_twam.preprocessing.rgb_motion_sequence import (
    RGBDCameraSequence,
    RGBMotionSequencePreprocessor,
)


class _PatchOutput:
    def __init__(self, tokens):
        self.tokens = tokens
        self.grid_size = tuple(tokens.shape[1:3])
        self.resized_size = (224, 224)


class _RecordingEncoder:
    """Use each frame's first RGB value as a visible DINO identity."""

    def __init__(self):
        self.calls = []

    def __call__(self, rgb):
        self.calls.append(rgb.detach().clone())
        if rgb.shape[1] in (1, 3, 4):
            identity = rgb[:, 0, 0, 0]
        else:
            identity = rgb[:, 0, 0, 0]
        offsets = torch.arange(4, device=rgb.device).reshape(1, 2, 2, 1)
        tokens = identity.to(torch.float32).reshape(-1, 1, 1, 1) + offsets
        return _PatchOutput(tokens)


class _RecordingDetector:
    max_tokens = None

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        dino_previous,
        dino_current,
        depth_previous,
        depth_current,
        pose_previous,
        pose_current,
        intrinsics_previous,
        *,
        wan_grid_size,
        camera_intrinsics_current,
        **kwargs,
    ):
        self.calls.append(
            {
                "dino_previous": dino_previous.detach().clone(),
                "dino_current": dino_current.detach().clone(),
                "depth_previous": depth_previous.detach().clone(),
                "depth_current": depth_current.detach().clone(),
                "pose_previous": pose_previous.detach().clone(),
                "pose_current": pose_current.detach().clone(),
                "intrinsics_previous": intrinsics_previous.detach().clone(),
                "intrinsics_current": camera_intrinsics_current.detach().clone(),
            }
        )
        batch = dino_current.shape[0]
        h, w = wan_grid_size
        mask = torch.zeros(batch, h, w, dtype=torch.bool, device=dino_current.device)
        score = torch.zeros(
            batch, h, w, dtype=torch.float32, device=dino_current.device
        )
        identity = dino_current[:, 0, 0, 0]
        for b in range(batch):
            # Camera A identities are <50: select local (row=1,col=0), score 5.
            # Camera B identities are >=50: select local (row=0,col=0), score 10.
            if identity[b] < 50:
                mask[b, 1, 0] = True
                score[b, 1, 0] = 5.0
            else:
                mask[b, 0, 0] = True
                score[b, 0, 0] = 10.0
        return SimpleNamespace(wan_motion_mask=mask, wan_motion_score=score)


class _TensorEncoder(_RecordingEncoder):
    def __call__(self, rgb):
        return super().__call__(rgb).tokens


def _sequence(values, *, depth_values=None):
    values = torch.as_tensor(values, dtype=torch.uint8)
    rgb = values[:, None, None, None].expand(-1, 3, 2, 2).clone()
    if depth_values is None:
        depth_values = values.to(torch.float32)
    depth_values = torch.as_tensor(depth_values, dtype=torch.float32)
    depth = depth_values[:, None, None].expand(-1, 2, 2).clone()
    poses = torch.eye(4).repeat(values.numel(), 1, 1)
    poses[:, 0, 3] = torch.arange(values.numel(), dtype=torch.float32)
    intrinsics = torch.eye(3).repeat(values.numel(), 1, 1)
    intrinsics[:, 0, 0] = torch.arange(values.numel(), dtype=torch.float32) + 1
    return RGBDCameraSequence(rgb, depth, poses, intrinsics)


def test_camera_sequence_rejects_unaligned_or_ambiguous_geometry():
    rgb = torch.zeros(2, 3, 4, 5, dtype=torch.uint8)
    pose = torch.eye(4).repeat(2, 1, 1)
    intrinsics = torch.eye(3).repeat(2, 1, 1)

    with pytest.raises(TypeError, match="calibrated floating-point"):
        RGBDCameraSequence(
            rgb, torch.ones(2, 4, 5, dtype=torch.uint16), pose, intrinsics
        )
    with pytest.raises(ValueError, match="pixel alignment"):
        RGBDCameraSequence(
            rgb, torch.ones(2, 3, 5), pose, intrinsics
        )

    bad_pose = pose.clone()
    bad_pose[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="camera_pose.*finite"):
        RGBDCameraSequence(
            rgb, torch.ones(2, 4, 5), bad_pose, intrinsics
        )

    bad_intrinsics = intrinsics.clone()
    bad_intrinsics[:, 0, 0] = 0
    with pytest.raises(ValueError, match="focal lengths"):
        RGBDCameraSequence(
            rgb, torch.ones(2, 4, 5), pose, bad_intrinsics
        )


def test_anchor_indices_drive_every_modality_and_first_frame_can_be_empty():
    encoder = _RecordingEncoder()
    detector = _RecordingDetector()
    processor = RGBMotionSequencePreprocessor(
        encoder,
        detector,
        max_tokens=3,
        wan_grid_size=(2, 2),
        first_frame_policy="empty",
    )
    output = processor(
        {"cam": _sequence(range(6))},
        anchor_indices=[1, 3, 5],
        world_time_ids=[101, 103, 105],
    )

    assert encoder.calls[0][:, 0, 0, 0].tolist() == [1, 3, 5]
    call = detector.calls[0]
    assert call["dino_previous"][:, 0, 0, 0].tolist() == [1.0, 3.0]
    assert call["dino_current"][:, 0, 0, 0].tolist() == [3.0, 5.0]
    assert call["depth_previous"][:, 0, 0].tolist() == [1.0, 3.0]
    assert call["depth_current"][:, 0, 0].tolist() == [3.0, 5.0]
    assert call["pose_current"][:, 0, 3].tolist() == [3.0, 5.0]
    assert call["intrinsics_current"][:, 0, 0].tolist() == [4.0, 6.0]

    assert output["motion_indices"].shape == (3, 3)
    assert output["motion_indices"][0].tolist() == [-1, -1, -1]
    assert output["motion_indices"][1:].tolist() == [[2, -1, -1], [2, -1, -1]]
    assert output["world_time_id"].tolist() == [
        [-1, -1, -1],
        [103, -1, -1],
        [105, -1, -1],
    ]
    assert output["observation_flag"].tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ]
    assert output["neoforce_features"].shape == (3, 3, 0)
    assert not output["tactile_valid"].any()


def test_multicamera_width_concat_happens_before_global_topk():
    processor = RGBMotionSequencePreprocessor(
        _RecordingEncoder(),
        _RecordingDetector(),
        max_tokens=2,
        wan_grid_size=(2, 2),
        first_frame_policy="empty",
        camera_keys=("a", "b"),
    )
    output = processor(
        {"b": _sequence([60, 61]), "a": _sequence([10, 11])},
        anchor_indices=[0, 1],
        world_time_ids=[7, 8],
    )

    # Camera order is explicitly a,b.  On the concatenated 2x4 grid:
    # b local (0,0) -> global 2; a local (1,0) -> global 4.  A naive per-camera
    # flat offset would incorrectly produce 4 and 2 respectively.
    assert output["motion_indices"][1].tolist() == [2, 4]
    assert output["motion_scores"][1].tolist() == [10.0, 5.0]
    assert output["dino_features"][1, :, 0].tolist() == [61.0, 13.0]
    assert output["spatial_grid_shape"] == (2, 4)
    assert output["camera_keys"] == ["a", "b"]
    assert output["provenance"]["camera_wan_grid_shapes"] == {
        "a": [2, 2],
        "b": [2, 2],
    }


def test_all_first_frame_is_complete_and_padding_stays_fixed_k():
    processor = RGBMotionSequencePreprocessor(
        _RecordingEncoder(),
        _RecordingDetector(),
        max_tokens=10,
        wan_grid_size=(2, 2),
        first_frame_policy="all",
    )
    output = processor(
        {"a": _sequence([10]), "b": _sequence([20])},
        anchor_indices=[0],
        world_time_ids=[42],
    )

    assert output["motion_indices"][0].tolist() == list(range(8)) + [-1, -1]
    assert output["motion_valid_mask"][0].tolist() == [True] * 8 + [False] * 2
    # Width concatenation gives row-major [a row, b row], not all of a then b.
    assert output["dino_features"][0, :8, 0].tolist() == [
        10.0,
        11.0,
        20.0,
        21.0,
        12.0,
        13.0,
        22.0,
        23.0,
    ]
    assert output["world_time_id"][0].tolist() == [42] * 8 + [-1, -1]


def test_all_first_frame_refuses_to_truncate_baseline():
    processor = RGBMotionSequencePreprocessor(
        _RecordingEncoder(),
        _RecordingDetector(),
        max_tokens=7,
        wan_grid_size=(2, 2),
        first_frame_policy="all",
    )
    with pytest.raises(ValueError, match=r"complete multi-camera grid \(8\)"):
        processor(
            {"a": _sequence([10]), "b": _sequence([20])},
            anchor_indices=[0],
            world_time_ids=[0],
        )


def test_require_previous_uses_explicit_segment_context():
    detector = _RecordingDetector()
    processor = RGBMotionSequencePreprocessor(
        _RecordingEncoder(),
        detector,
        max_tokens=2,
        wan_grid_size=(2, 2),
        first_frame_policy="require_previous",
    )
    sequence = _sequence([10, 11, 12])
    with pytest.raises(ValueError, match="needs previous_frames"):
        processor(
            {"cam": sequence},
            anchor_indices=[0, 2],
            world_time_ids=[5, 6],
        )

    previous = {
        "cam": {
            "rgb": torch.full((3, 2, 2), 9, dtype=torch.uint8),
            "depth": torch.full((2, 2), 9.0),
            "camera_pose": torch.eye(4),
            "camera_intrinsics": torch.eye(3),
        }
    }
    output = processor(
        {"cam": sequence},
        anchor_indices=[0, 2],
        world_time_ids=[5, 6],
        previous_frames=previous,
        observation_flag=[0, 1],
    )
    call = detector.calls[-1]
    assert call["dino_previous"][:, 0, 0, 0].tolist() == [9.0, 10.0]
    assert call["dino_current"][:, 0, 0, 0].tolist() == [10.0, 12.0]
    assert call["depth_previous"][:, 0, 0].tolist() == [9.0, 10.0]
    assert output["observation_flag"][:, 0].tolist() == [0, 1]
    assert output["provenance"]["anchor_indices"] == [0, 2]
    assert output["provenance"]["world_time_ids"] == [5, 6]


def test_world_time_and_anchor_validation_is_strict():
    processor = RGBMotionSequencePreprocessor(
        _RecordingEncoder(),
        _RecordingDetector(),
        max_tokens=2,
        first_frame_policy="empty",
    )
    sequence = _sequence([0, 1, 2])
    with pytest.raises(ValueError, match="world_time_ids must have 2 entries"):
        processor({"cam": sequence}, anchor_indices=[0, 2], world_time_ids=[10])
    with pytest.raises(ValueError, match="strictly increasing"):
        processor({"cam": sequence}, anchor_indices=[1, 1], world_time_ids=[10, 11])


def test_detector_cannot_apply_a_private_per_camera_budget():
    detector = _RecordingDetector()
    detector.max_tokens = 1
    with pytest.raises(ValueError, match="detector.max_tokens must be None"):
        RGBMotionSequencePreprocessor(_RecordingEncoder(), detector, max_tokens=2)


@pytest.mark.parametrize("channels", [1, 4])
def test_camera_sequence_requires_true_rgb(channels):
    with pytest.raises(ValueError, match="exactly 3 channels"):
        RGBDCameraSequence(
            rgb=torch.zeros(2, channels, 4, 4),
            depth=torch.ones(2, 4, 4),
            camera_pose=torch.eye(4),
            camera_intrinsics=torch.eye(3),
        )

    with pytest.raises(ValueError, match="unambiguously"):
        RGBDCameraSequence(
            rgb=torch.zeros(2, 3, 4, 3),
            depth=torch.ones(2, 4, 3),
            camera_pose=torch.eye(4),
            camera_intrinsics=torch.eye(3),
        )


def test_real_detector_interface_runs_through_sequence_preprocessor():
    detector = EgoMotionCompensatedMotionDetector(
        depth_threshold=0.01,
        dino_threshold=0.1,
        dilation_radius=0,
        max_tokens=None,
    )
    processor = RGBMotionSequencePreprocessor(
        _RecordingEncoder(),
        detector,
        max_tokens=4,
        wan_grid_size=(2, 2),
        first_frame_policy="empty",
    )
    sequence = _sequence([10, 10], depth_values=[1.0, 1.0])
    sequence.camera_pose.zero_()
    sequence.camera_pose[:, range(4), range(4)] = 1
    sequence.camera_intrinsics.zero_()
    sequence.camera_intrinsics[:, range(3), range(3)] = 1

    output = processor({"cam": sequence}, anchor_indices=[0, 1], world_time_ids=[0, 1])
    assert output["motion_indices"].shape == (2, 4)
    assert not output["motion_valid_mask"].any()


def test_injected_encoder_may_return_a_bare_tensor():
    processor = RGBMotionSequencePreprocessor(
        _TensorEncoder(),
        _RecordingDetector(),
        max_tokens=4,
        wan_grid_size=(2, 2),
        first_frame_policy="all",
    )
    output = processor({"cam": _sequence([10])}, anchor_indices=[0], world_time_ids=[0])
    assert output["dino_features"].shape == (1, 4, 1)
    assert output["motion_indices"].tolist() == [[0, 1, 2, 3]]
