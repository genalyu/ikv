from dataclasses import replace

import pytest
import torch

from n0_twam.models.rgb_motion import (
    EgoMotionCompensatedMotionDetector,
    MotionDetectionResult,
    SparsePatchGather,
    SparsePatchScatter,
    TokenIndex,
    gather_grid_features,
    motion_result_to_sidecar,
    patchify_latents,
    pool_dino_to_grid,
    unpatchify_latents,
)
from n0_twam.utils.geometry import (
    backproject_pixels,
    project_points,
    relative_camera_transform,
    transform_points,
)


def _camera_matrix(height: int, width: int, focal: float = 20.0) -> torch.Tensor:
    return torch.tensor(
        [
            [focal, 0.0, (width - 1) / 2],
            [0.0, focal, (height - 1) / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_geometry_backproject_transform_and_project() -> None:
    pixels = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])
    depth = torch.tensor([[2.0, 4.0]])
    intrinsics = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    points = backproject_pixels(pixels, depth, intrinsics)
    expected = torch.tensor([[[0.0, 1.0, 2.0], [4.0, 0.0, 4.0]]])
    torch.testing.assert_close(points, expected)

    transform = torch.eye(4)
    transform[0, 3] = 1.0
    transformed = transform_points(points, transform)
    projected, z, valid = project_points(transformed, intrinsics)
    torch.testing.assert_close(z, depth)
    assert valid.all()
    torch.testing.assert_close(projected, torch.tensor([[[2.0, 2.0], [3.5, 1.0]]]))


def test_relative_camera_transform_pose_conventions() -> None:
    previous_world_from_camera = torch.eye(4)
    current_world_from_camera = torch.eye(4)
    current_world_from_camera[0, 3] = 0.25
    previous_from_current = relative_camera_transform(
        previous_world_from_camera, current_world_from_camera
    )
    torch.testing.assert_close(previous_from_current[0, 0, 3], torch.tensor(0.25))

    previous_camera_from_world = torch.linalg.inv(previous_world_from_camera)
    current_camera_from_world = torch.linalg.inv(current_world_from_camera)
    alternate = relative_camera_transform(
        previous_camera_from_world,
        current_camera_from_world,
        convention="camera_from_world",
    )
    torch.testing.assert_close(alternate, previous_from_current)


def test_motion_detector_identity_is_static_and_maps_grid() -> None:
    generator = torch.Generator().manual_seed(3)
    dino = torch.randn(2, 4, 4, 6, generator=generator)
    depth = torch.ones(2, 4, 4)
    pose = torch.eye(4).expand(2, -1, -1).clone()
    detector = EgoMotionCompensatedMotionDetector(
        depth_threshold=0.01,
        dino_threshold=0.1,
        dilation_radius=0,
    )
    result = detector(
        dino,
        dino.clone(),
        depth,
        depth.clone(),
        pose,
        pose.clone(),
        _camera_matrix(4, 4),
        wan_grid_size=(2, 2),
    )

    assert result.dino_motion_score.shape == (2, 4, 4)
    assert result.wan_motion_score.shape == (2, 2, 2)
    assert result.wan_dino_features.shape == (2, 2, 2, 6)
    assert not result.dino_motion_mask.any()
    assert not result.wan_motion_mask.any()
    assert result.wan_moving_indices.shape == (2, 0)


def test_motion_detector_compensates_lateral_camera_motion() -> None:
    # An infinite fronto-parallel plane has constant depth and appearance. Moving
    # the camera laterally changes correspondence coordinates, but not the scene.
    dino = torch.ones(1, 6, 6, 4)
    depth = torch.ones(1, 24, 24)
    pose_previous = torch.eye(4)
    pose_current = torch.eye(4)
    pose_current[0, 3] = 0.05
    detector = EgoMotionCompensatedMotionDetector(
        depth_threshold=0.01,
        dino_threshold=0.1,
        dilation_radius=0,
    )
    result = detector(
        dino,
        dino,
        depth,
        depth,
        pose_previous,
        pose_current,
        _camera_matrix(24, 24, focal=40.0),
    )

    # Some boundary correspondences leave the previous view; those are unknown,
    # not falsely labelled as moving.
    assert not result.dino_motion_mask.any()
    assert result.dino_correspondence_valid.any()
    assert (~result.dino_correspondence_valid).any()


def test_motion_detector_finds_dino_and_depth_changes_with_topk() -> None:
    dino_previous = torch.zeros(1, 4, 4, 3)
    dino_previous[..., 0] = 1.0
    dino_current = dino_previous.clone()
    dino_current[0, 1, 1] = torch.tensor([-1.0, 0.0, 0.0])
    dino_current[0, 3, 3] = torch.tensor([0.0, 1.0, 0.0])
    depth_previous = torch.ones(1, 4, 4)
    depth_current = depth_previous.clone()
    depth_current[0, 2, 2] = 1.5
    detector = EgoMotionCompensatedMotionDetector(
        depth_threshold=0.05,
        dino_threshold=0.2,
        dilation_radius=0,
        max_tokens=2,
    )
    result = detector(
        dino_previous,
        dino_current,
        depth_previous,
        depth_current,
        torch.eye(4),
        torch.eye(4),
        _camera_matrix(4, 4),
    )

    assert result.dino_motion_mask[0, 1, 1]
    assert result.dino_motion_mask[0, 2, 2]
    assert result.dino_motion_mask[0, 3, 3]
    assert result.wan_indices_valid.sum().item() == 2
    assert result.wan_motion_mask.sum().item() == 2
    assert result.motion_mask is result.wan_motion_mask
    assert result.moving_patch_indices is result.wan_moving_indices
    assert result.gather_current_dino().shape == (1, 2, 3)
    chosen = set(result.wan_moving_indices[0].tolist())
    # The reversed DINO vector (index 5) has a larger score than the orthogonal
    # feature (index 15); the large depth residual at index 10 is also retained.
    assert chosen == {5, 10}


def test_flattened_dino_and_pooling_to_wan_grid() -> None:
    dino = torch.arange(1 * 4 * 4 * 2, dtype=torch.float32).reshape(1, 16, 2)
    pooled = pool_dino_to_grid(dino, (2, 2), source_grid_size=(4, 4))
    assert pooled.shape == (1, 2, 2, 2)
    source = dino.reshape(1, 4, 4, 2)
    torch.testing.assert_close(pooled[0, 0, 0], source[0, :2, :2].mean((0, 1)))


def test_patchify_is_exactly_invertible() -> None:
    latents = torch.arange(2 * 3 * 2 * 4 * 6, dtype=torch.float32).reshape(
        2, 3, 2, 4, 6
    )
    patches, grid = patchify_latents(latents, (1, 2, 2))
    assert grid == (2, 2, 3)
    assert patches.shape == (2, 12, 12)
    restored = unpatchify_latents(patches, grid, (1, 2, 2), channels=3)
    torch.testing.assert_close(restored, latents)


def test_sparse_gather_and_scatter_are_batch_safe() -> None:
    latents = torch.arange(2 * 2 * 2 * 4 * 4, dtype=torch.float32).reshape(
        2, 2, 2, 4, 4
    )
    # Patch grid is (F=2,H=2,W=2). Batch 0 selects three patches, batch 1 one.
    mask = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
    mask[0].reshape(-1)[torch.tensor([0, 3, 7])] = True
    mask[1].reshape(-1)[torch.tensor([5])] = True
    sparse = SparsePatchGather((1, 2, 2))(latents, motion_mask=mask)

    assert sparse.values.shape == (2, 3, 8)
    assert sparse.indices.tolist() == [[0, 3, 7], [5, -1, -1]]
    assert sparse.valid_mask.tolist() == [[True, True, True], [True, False, False]]
    assert not sparse.values[1, 1:].any()

    replacement = sparse.values + 1000.0
    scattered = SparsePatchScatter()(sparse, base_latents=latents, values=replacement)
    original_patches, _ = patchify_latents(latents)
    output_patches, _ = patchify_latents(scattered)
    for batch_index, selected in enumerate(([0, 3, 7], [5])):
        selected_tensor = torch.tensor(selected)
        torch.testing.assert_close(
            output_patches[batch_index, selected_tensor],
            original_patches[batch_index, selected_tensor] + 1000.0,
        )
        unselected = torch.ones(8, dtype=torch.bool)
        unselected[selected_tensor] = False
        torch.testing.assert_close(
            output_patches[batch_index, unselected],
            original_patches[batch_index, unselected],
        )


def test_sparse_scatter_without_base_leaves_unselected_patches_zero() -> None:
    latents = torch.randn(1, 2, 1, 4, 4)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    mask[0, 0, 1, 0] = True
    sparse = SparsePatchGather()(latents, motion_mask=mask)
    scattered = SparsePatchScatter()(sparse)
    patches, _ = patchify_latents(scattered)
    assert torch.count_nonzero(patches[0, [0, 1, 3]]) == 0
    torch.testing.assert_close(patches[0, 2], sparse.values[0, 0])


def test_gather_grid_features_uses_same_sparse_addressing() -> None:
    features = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3)
    indices = torch.tensor([[3, 0], [1, -1]])
    valid = torch.tensor([[True, True], [True, False]])
    result = gather_grid_features(features, indices, valid)
    torch.testing.assert_close(result[0, 0], features[0, 1, 1])
    torch.testing.assert_close(result[0, 1], features[0, 0, 0])
    torch.testing.assert_close(result[1, 0], features[1, 0, 1])
    assert not result[1, 1].any()


def test_token_index_stays_factorized_and_validates_presence() -> None:
    index = TokenIndex(
        world_time_id=torch.tensor([[7, 8, 0], [7, 8, 0]]),
        dino=torch.randn(2, 3, 5),
        neoforce=torch.randn(2, 3, 4),
        observation_flag=torch.tensor([[1, 0, 0], [1, 1, 0]]),
        visual_valid=torch.tensor([[True, False, False], [True, True, False]]),
        tactile_valid=torch.tensor([[False, True, False], [True, False, False]]),
    )
    token_valid = torch.tensor([[True, True, False], [True, True, False]])
    index.validate_presence(token_valid)
    assert index.observed.tolist() == [[True, False, False], [True, True, False]]
    assert index.predicted.tolist() == [[False, True, True], [False, False, True]]

    bad_valid = token_valid.clone()
    bad_valid[0, 2] = True
    with pytest.raises(ValueError, match="at least one"):
        index.validate_presence(bad_valid)


def test_token_index_supports_rgb_only_zero_width_neoforce() -> None:
    rgb_only = TokenIndex(
        world_time_id=torch.tensor([[4, 5]]),
        dino=torch.randn(1, 2, 6),
        neoforce=torch.empty(1, 2, 0),
        observation_flag=torch.tensor([[1, 0]]),
        visual_valid=torch.tensor([[True, True]]),
        tactile_valid=torch.tensor([[False, False]]),
    )
    rgb_only.validate_presence()
    assert rgb_only.neoforce.shape == (1, 2, 0)

    tactile_only = TokenIndex(
        world_time_id=torch.tensor([[4]]),
        dino=torch.empty(1, 1, 0),
        neoforce=torch.randn(1, 1, 4),
        observation_flag=torch.tensor([[1]]),
        visual_valid=torch.tensor([[False]]),
        tactile_valid=torch.tensor([[True]]),
    )
    tactile_only.validate_presence()

    with pytest.raises(ValueError, match="tactile_valid must be all False"):
        TokenIndex(
            world_time_id=torch.tensor([[4]]),
            dino=torch.randn(1, 1, 6),
            neoforce=torch.empty(1, 1, 0),
            observation_flag=torch.tensor([[1]]),
            visual_valid=torch.tensor([[True]]),
            tactile_valid=torch.tensor([[True]]),
        )

    with pytest.raises(ValueError, match="cannot both be zero"):
        TokenIndex(
            world_time_id=torch.tensor([[4]]),
            dino=torch.empty(1, 1, 0),
            neoforce=torch.empty(1, 1, 0),
            observation_flag=torch.tensor([[1]]),
            visual_valid=torch.tensor([[False]]),
            tactile_valid=torch.tensor([[False]]),
        )


def _sidecar_motion_result() -> MotionDetectionResult:
    scores = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.9]],
            [[0.4, 0.8], [0.6, 0.7]],
        ]
    )
    indices = torch.tensor([[3, 0], [1, -1]])
    valid = torch.tensor([[True, True], [True, False]])
    mask = torch.tensor(
        [
            [[True, False], [False, True]],
            [[False, True], [False, False]],
        ]
    )
    dino = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3)
    zeros = torch.zeros(2, 2, 2)
    return MotionDetectionResult(
        dino_motion_score=zeros,
        dino_motion_mask=mask,
        dino_correspondence_valid=torch.ones_like(mask),
        depth_residual=zeros,
        dino_distance=zeros,
        wan_motion_score=scores,
        wan_motion_mask=mask,
        wan_moving_indices=indices,
        wan_indices_valid=valid,
        wan_dino_features=dino,
    )


def test_motion_result_to_sidecar_rgb_only_maps_batch_to_frames() -> None:
    result = _sidecar_motion_result()
    sidecar = motion_result_to_sidecar(
        result,
        world_time_id=torch.tensor([10, 11]),
        observation_flag=torch.tensor([1, 0]),
    )

    assert set(sidecar) == {
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
    assert sidecar["motion_indices"].tolist() == [[3, 0], [1, -1]]
    torch.testing.assert_close(
        sidecar["motion_scores"], torch.tensor([[0.9, 0.1], [0.8, 0.0]])
    )
    assert sidecar["world_time_id"].tolist() == [[10, 10], [11, -1]]
    assert sidecar["observation_flag"].tolist() == [[1, 1], [0, 0]]
    assert sidecar["visual_valid"].equal(sidecar["motion_valid_mask"])
    assert not sidecar["tactile_valid"].any()
    assert sidecar["neoforce_features"].shape == (2, 2, 0)
    torch.testing.assert_close(
        sidecar["dino_features"][0],
        result.wan_dino_features[0].reshape(4, 3)[torch.tensor([3, 0])],
    )
    assert not sidecar["dino_features"][1, 1].any()


def test_motion_result_to_sidecar_gathers_neoforce_with_explicit_presence() -> None:
    result = _sidecar_motion_result()
    neoforce_grid = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2)
    tactile_grid = torch.tensor(
        [
            [[False, False], [False, True]],
            [[False, True], [False, False]],
        ]
    )
    sidecar = motion_result_to_sidecar(
        result,
        world_time_id=5,
        observation_flag=True,
        neoforce_features=neoforce_grid,
        tactile_valid=tactile_grid,
    )

    assert sidecar["tactile_valid"].tolist() == [[True, False], [True, False]]
    torch.testing.assert_close(
        sidecar["neoforce_features"][0, 0], neoforce_grid[0, 1, 1]
    )
    torch.testing.assert_close(
        sidecar["neoforce_features"][1, 0], neoforce_grid[1, 0, 1]
    )
    assert not sidecar["neoforce_features"][1, 1].any()


def test_motion_result_to_sidecar_rejects_ambiguous_or_inconsistent_metadata() -> None:
    result = _sidecar_motion_result()
    with pytest.raises(ValueError, match="tactile_valid is required"):
        motion_result_to_sidecar(
            result,
            world_time_id=torch.tensor([0, 1]),
            observation_flag=1,
            neoforce_features=torch.zeros(2, 2, 4),
        )
    with pytest.raises(ValueError, match="only 0 or 1"):
        motion_result_to_sidecar(
            result,
            world_time_id=torch.tensor([0, 1]),
            observation_flag=2,
        )
    with pytest.raises(ValueError, match="disagrees"):
        motion_result_to_sidecar(
            replace(result, wan_motion_mask=torch.zeros_like(result.wan_motion_mask)),
            world_time_id=torch.tensor([0, 1]),
            observation_flag=1,
        )


def test_shape_validation_reports_wrong_motion_grid() -> None:
    latents = torch.zeros(1, 2, 1, 4, 4)
    with pytest.raises(ValueError, match="motion_mask must have shape"):
        SparsePatchGather()(latents, motion_mask=torch.zeros(1, 3, 3, dtype=torch.bool))

    detector = EgoMotionCompensatedMotionDetector(dilation_radius=0)
    with pytest.raises(ValueError, match="grid_size is required"):
        detector(
            torch.zeros(1, 16, 3),
            torch.zeros(1, 16, 3),
            torch.ones(1, 4, 4),
            torch.ones(1, 4, 4),
            torch.eye(4),
            torch.eye(4),
            _camera_matrix(4, 4),
        )
