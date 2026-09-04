"""Integration coverage for sparse RGB-motion wiring in model and trainer.

These tests intentionally stop before the transformer backbone: a zero-block,
CPU-only WAN model is sufficient to verify patch addressing/projection, while
the Trainer loss is exercised through a lightweight instance and scheduler.
That keeps the suite independent of FlexAttention compilation and CUDA.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

from einops import rearrange
import pytest
import torch
import torch.nn.functional as F

from n0_twam.models.model import WanTransformer3DModel
from n0_twam.utils.utils import data_seq_to_patch


def _load_trainer_class():
    """Load Trainer without importing optional LeRobot training dependencies."""

    module_path = Path(__file__).parents[1] / "n0_twam" / "train.py"

    configs = types.ModuleType("configs")
    configs.TWAM_CONFIGS = {}

    distributed = types.ModuleType("distributed")
    distributed.__path__ = []
    distributed_fsdp = types.ModuleType("distributed.fsdp")
    distributed_fsdp.shard_model = lambda *args, **kwargs: None
    distributed_fsdp.apply_ac = lambda *args, **kwargs: None
    distributed_util = types.ModuleType("distributed.util")
    distributed_util._configure_model = lambda model, **kwargs: model
    distributed_util.init_distributed = lambda *args, **kwargs: None
    distributed_util.dist_mean = lambda value: value
    distributed_util.dist_max = lambda value: value

    models = types.ModuleType("models")
    models.__path__ = []
    models_utils = types.ModuleType("models.utils")
    models_utils.load_transformer = lambda *args, **kwargs: None
    models_utils.load_mot_transformer = lambda *args, **kwargs: None
    models_utils.load_mot_checkpoint = lambda *args, **kwargs: None

    utils = types.ModuleType("utils")
    utils.init_logger = lambda *args, **kwargs: None
    utils.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    utils.get_mesh_id = lambda *args, **kwargs: None
    utils.sample_timestep_id = lambda *args, **kwargs: None
    utils.data_seq_to_patch = data_seq_to_patch
    utils.warmup_constant_lambda = lambda *args, **kwargs: 1.0
    utils.FlowMatchScheduler = object

    dataset = types.ModuleType("dataset")
    dataset.MultiLatentLeRobotDataset = object
    dataset.BucketedDistributedBatchSampler = object

    wandb = types.ModuleType("wandb")
    wandb.login = lambda *args, **kwargs: None
    wandb.init = lambda *args, **kwargs: None

    stubs = {
        "configs": configs,
        "distributed": distributed,
        "distributed.fsdp": distributed_fsdp,
        "distributed.util": distributed_util,
        "models": models,
        "models.utils": models_utils,
        "utils": utils,
        "dataset": dataset,
        "wandb": wandb,
    }
    spec = importlib.util.spec_from_file_location(
        "_rgb_motion_train_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.Trainer


Trainer = _load_trainer_class()


class _StubScheduler:
    @staticmethod
    def training_weight(timesteps: torch.Tensor) -> torch.Tensor:
        # Deliberately non-uniform so the test catches wrong frame addressing.
        return timesteps.float() + 0.5


def test_rgb_motion_training_requires_mot_experts() -> None:
    Trainer._validate_rgb_motion_training_config(
        types.SimpleNamespace(use_rgb_motion_tokens=False, use_mot=False)
    )
    Trainer._validate_rgb_motion_training_config(
        types.SimpleNamespace(use_rgb_motion_tokens=True, use_mot=True)
    )
    with pytest.raises(ValueError, match="requires use_mot=True"):
        Trainer._validate_rgb_motion_training_config(
            types.SimpleNamespace(use_rgb_motion_tokens=True, use_mot=False)
        )


def _tiny_model(*, sparse: bool) -> WanTransformer3DModel:
    torch.manual_seed(7)
    return (
        WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=1,
            attention_head_dim=4,
            in_channels=2,
            out_channels=2,
            action_dim=3,
            text_dim=8,
            freq_dim=4,
            ffn_dim=8,
            num_layers=0,
            rope_max_seq_len=16,
            attn_mode="torch",
            use_local_tactile=False,
            use_rgb_motion_tokens=sparse,
            rgb_motion_require_index=True,
        )
        .cpu()
        .eval()
    )


def _semantic_sidecar(
    indices: torch.Tensor, valid: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Build a complete, factorized {t,DINO,NeoForce,obs/pred} index."""

    batch, frames, per_frame = indices.shape
    world_time = torch.arange(frames)[None, :, None].expand(batch, -1, per_frame) + 10
    dino = torch.arange(batch * frames * per_frame * 3, dtype=torch.float32).reshape(
        batch, frames, per_frame, 3
    )
    return {
        "world_time_id": world_time,
        "dino_features": dino,
        # Width zero is the explicit representation of absent NeoForce data.
        "neoforce_features": torch.empty(batch, frames, per_frame, 0),
        "observation_flag": torch.ones_like(indices),
        "visual_valid": valid.clone(),
        "tactile_valid": torch.zeros_like(valid),
    }


def _motion_input(
    latents: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if valid is None:
        valid = indices >= 0
    result = {
        "noisy_latents": latents,
        "rgb_motion_patch_indices": indices,
        "rgb_motion_valid_mask": valid,
    }
    result.update(_semantic_sidecar(indices, valid))
    return result


def test_raw_wan_patches_are_gathered_before_the_unchanged_projection() -> None:
    model = _tiny_model(sparse=True)
    latents = torch.arange(2 * 2 * 2 * 4 * 4, dtype=torch.float32).reshape(
        2, 2, 2, 4, 4
    )
    # Local indices address the 2x2 spatial grid independently in each frame.
    local = torch.tensor(
        [
            [[0, 3], [1, 2]],
            [[1, 2], [0, 3]],
        ]
    )
    layout = model._rgb_motion_layout(_motion_input(latents, local), latents.shape)

    raw_dense = rearrange(
        latents,
        "b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)",
        pt=1,
        ph=2,
        pw=2,
    )
    expected_raw = torch.gather(
        raw_dense,
        1,
        layout["indices"][..., None].expand(-1, -1, raw_dense.shape[-1]),
    )
    dense_projected = model._input_embed(latents, input_type="latent")
    expected_projected = torch.gather(
        dense_projected,
        1,
        layout["indices"][..., None].expand(-1, -1, dense_projected.shape[-1]),
    )

    projection_before = {
        name: tensor.detach().clone()
        for name, tensor in model.patch_embedding_mlp.state_dict().items()
    }
    projection_inputs: list[torch.Tensor] = []
    hook = model.patch_embedding_mlp.register_forward_pre_hook(
        lambda _module, args: projection_inputs.append(args[0].detach().clone())
    )
    try:
        sparse_projected = model._input_embed(
            latents, input_type="latent", motion_layout=layout
        )
    finally:
        hook.remove()

    assert len(projection_inputs) == 1
    torch.testing.assert_close(projection_inputs[0], expected_raw)
    torch.testing.assert_close(sparse_projected, expected_projected)
    for name, tensor in model.patch_embedding_mlp.state_dict().items():
        torch.testing.assert_close(tensor, projection_before[name])


def test_local_b_f_k_layout_normalizes_indices_frames_and_padding() -> None:
    model = _tiny_model(sparse=True)
    latents = torch.zeros(2, 2, 2, 4, 4)
    local = torch.tensor(
        [
            [[3, -1, -1], [0, 2, -1]],
            [[1, 0, -1], [3, -1, -1]],
        ]
    )
    valid = local >= 0
    sidecar = _motion_input(latents, local, valid)
    # Non-canonical padding values must be discarded during normalization.
    sidecar["world_time_id"] = sidecar["world_time_id"].masked_fill(~valid, 999)
    sidecar["dino_features"] = sidecar["dino_features"].masked_fill(
        ~valid[..., None], 999.0
    )
    sidecar["observation_flag"] = sidecar["observation_flag"].masked_fill(~valid, 9)

    layout = model._rgb_motion_layout(sidecar, latents.shape)

    # Each frame owns four spatial tokens. A padded -1 is converted to a safe
    # per-frame gather address, while valid_mask remains the source of truth.
    assert layout["indices"].tolist() == [
        [3, 0, 0, 4, 6, 4],
        [1, 0, 0, 7, 4, 4],
    ]
    assert layout["frame_ids"].tolist() == [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]
    assert torch.equal(layout["valid_mask"], valid.reshape(2, 6))
    assert layout["grid_shape"] == (2, 2, 2)

    semantic = layout["semantic_index"]
    assert semantic is not None
    padding = ~layout["valid_mask"]
    assert torch.equal(
        semantic["world_time_id"][padding],
        torch.full_like(semantic["world_time_id"][padding], -1),
    )
    assert not semantic["dino"][padding].any()
    assert not semantic["observation_flag"][padding].any()
    assert not semantic["visual_valid"][padding].any()
    assert not semantic["tactile_valid"].any()

    raw_tokens = torch.arange(2 * 8 * 3).reshape(2, 8, 3)
    gathered = model._gather_rgb_motion_tokens(raw_tokens, layout)
    assert not gathered[padding].any()


@pytest.mark.parametrize("flattened", [False, True])
def test_invalid_stale_positive_motion_address_is_safe(flattened: bool) -> None:
    model = _tiny_model(sparse=True)
    latents = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(
        1, 2, 1, 4, 4
    )
    valid = torch.tensor([[[True, False]]])
    sidecar = _motion_input(latents, torch.tensor([[[0, 99]]]), valid)
    if flattened:
        sidecar["rgb_motion_indices"] = sidecar.pop(
            "rgb_motion_patch_indices"
        ).reshape(1, -1)
        sidecar["rgb_motion_valid_mask"] = valid.reshape(1, -1)
        for name in (
            "world_time_id",
            "observation_flag",
            "visual_valid",
            "tactile_valid",
        ):
            sidecar[name] = sidecar[name].reshape(1, -1)
        for name in ("dino_features", "neoforce_features"):
            sidecar[name] = sidecar[name].reshape(
                1, 2, sidecar[name].shape[-1]
            )

    layout = model._rgb_motion_layout(sidecar, latents.shape)
    assert layout["indices"].tolist() == [[0, 0]]
    projected = model._input_embed(latents, motion_layout=layout)
    assert projected.shape[:2] == (1, 2)


@pytest.mark.parametrize("flattened", [False, True])
def test_valid_out_of_range_motion_address_is_rejected(flattened: bool) -> None:
    model = _tiny_model(sparse=True)
    latents = torch.zeros(1, 2, 1, 4, 4)
    valid = torch.tensor([[[True]]])
    sidecar = _motion_input(latents, torch.tensor([[[99]]]), valid)
    if flattened:
        sidecar["rgb_motion_indices"] = sidecar.pop(
            "rgb_motion_patch_indices"
        ).reshape(1, -1)
        sidecar["rgb_motion_valid_mask"] = valid.reshape(1, -1)
        for name in (
            "world_time_id",
            "observation_flag",
            "visual_valid",
            "tactile_valid",
        ):
            sidecar[name] = sidecar[name].reshape(1, -1)
        for name in ("dino_features", "neoforce_features"):
            sidecar[name] = sidecar[name].reshape(
                1, 1, sidecar[name].shape[-1]
            )

    with pytest.raises(IndexError, match="out of range"):
        model._rgb_motion_layout(sidecar, latents.shape)


def test_direct_model_rejects_nonfinite_present_semantic_features() -> None:
    model = _tiny_model(sparse=True)
    latents = torch.zeros(1, 2, 1, 4, 4)
    sidecar = _motion_input(latents, torch.tensor([[[0]]]))
    sidecar["dino_features"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="dino_features.*finite"):
        model._rgb_motion_layout(sidecar, latents.shape)

    sidecar = _motion_input(latents, torch.tensor([[[0]]]))
    sidecar["neoforce_features"] = torch.full((1, 1, 1, 3), float("inf"))
    sidecar["tactile_valid"] = torch.ones(1, 1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="neoforce_features.*finite"):
        model._rgb_motion_layout(sidecar, latents.shape)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("rgb_motion_valid_mask", torch.tensor([[[2]]]), "only 0 or 1"),
        ("visual_valid", torch.tensor([[[2]]]), "only 0 or 1"),
        ("observation_flag", torch.tensor([[[0.5]]]), "only 0 or 1"),
        ("world_time_id", torch.tensor([[[1.5]]]), "integer values"),
    ],
)
def test_direct_model_rejects_ambiguous_index_metadata(field, value, error) -> None:
    model = _tiny_model(sparse=True)
    latents = torch.zeros(1, 2, 1, 4, 4)
    sidecar = _motion_input(latents, torch.tensor([[[0]]]))
    sidecar[field] = value
    with pytest.raises((TypeError, ValueError), match=error):
        model._rgb_motion_layout(sidecar, latents.shape)


def test_direct_model_rejects_fractional_or_duplicate_index_layouts() -> None:
    model = _tiny_model(sparse=True)
    latents = torch.zeros(1, 2, 1, 4, 4)
    sidecar = _motion_input(latents, torch.tensor([[[0]]]))
    sidecar["rgb_motion_patch_indices"] = torch.tensor([[[0.5]]])
    with pytest.raises(ValueError, match="integer values"):
        model._rgb_motion_layout(sidecar, latents.shape)

    sidecar = _motion_input(latents, torch.tensor([[[0, 0]]]))
    with pytest.raises(ValueError, match="duplicate valid addresses"):
        model._rgb_motion_layout(sidecar, latents.shape)

    sidecar = _motion_input(latents, torch.tensor([[[0]]]))
    sidecar["rgb_motion_indices"] = torch.tensor([[0]])
    with pytest.raises(KeyError, match="exactly one"):
        model._rgb_motion_layout(sidecar, latents.shape)


@pytest.mark.parametrize("flattened", [False, True])
def test_sparse_loss_safely_ignores_invalid_stale_positive_address(
    flattened: bool,
) -> None:
    trainer = object.__new__(Trainer)
    trainer.patch_size = (1, 2, 2)
    trainer.train_scheduler_latent = _StubScheduler()
    target = torch.zeros(1, 2, 1, 4, 4)
    valid = torch.tensor([[[True, False]]])
    indices = torch.tensor([[[0, 99]]])
    latent_dict = {
        "targets": target,
        "timesteps": torch.zeros(1, 1),
        "rgb_motion_valid_mask": valid,
    }
    if flattened:
        latent_dict["rgb_motion_indices"] = indices.reshape(1, -1)
        latent_dict["rgb_motion_valid_mask"] = valid.reshape(1, -1)
    else:
        latent_dict["rgb_motion_patch_indices"] = indices
    prediction = torch.zeros(1, 2 * 4, 2)

    loss = trainer._compute_latent_loss(
        {"latent_dict": latent_dict}, prediction
    )
    assert loss.item() == 0.0


@pytest.mark.parametrize("flattened", [False, True])
def test_sparse_loss_rejects_valid_out_of_range_address(flattened: bool) -> None:
    trainer = object.__new__(Trainer)
    trainer.patch_size = (1, 2, 2)
    trainer.train_scheduler_latent = _StubScheduler()
    valid = torch.tensor([[[True]]])
    indices = torch.tensor([[[99]]])
    latent_dict = {
        "targets": torch.zeros(1, 2, 1, 4, 4),
        "timesteps": torch.zeros(1, 1),
        "rgb_motion_valid_mask": valid,
    }
    if flattened:
        latent_dict["rgb_motion_indices"] = indices.reshape(1, -1)
        latent_dict["rgb_motion_valid_mask"] = valid.reshape(1, -1)
    else:
        latent_dict["rgb_motion_patch_indices"] = indices

    with pytest.raises(IndexError, match="out of range"):
        trainer._compute_latent_loss(
            {"latent_dict": latent_dict}, torch.zeros(1, 4, 2)
        )


def test_sparse_tactile_frame_start_ignores_zeroed_grid_padding() -> None:
    model = _tiny_model(sparse=True)
    latents = torch.zeros(1, 2, 2, 4, 4)
    dense_grid = torch.zeros(1, 4, 8, dtype=torch.long)
    dense_grid[0, 0] = torch.tensor([9, 9, 9, 9, 10, 10, 10, 10])

    # Frame 0 contains only padding; the sole real motion token is in frame 1.
    local = torch.tensor([[[-1, -1], [2, -1]]])
    layout = model._rgb_motion_layout(_motion_input(latents, local), latents.shape)
    gathered_grid = model._gather_rgb_motion_grid(dense_grid, layout)
    assert gathered_grid[0, 0].tolist() == [0, 0, 10, 0]
    assert model._inference_video_frame_start(dense_grid, gathered_grid, layout) == 10

    # With no valid motion tokens at all, retain the streaming chunk's actual
    # dense origin rather than the padding sentinel zero.
    empty_local = torch.full((1, 2, 2), -1)
    empty_layout = model._rgb_motion_layout(
        _motion_input(latents, empty_local), latents.shape
    )
    empty_grid = model._gather_rgb_motion_grid(dense_grid, empty_layout)
    assert not empty_grid.any()
    assert model._inference_video_frame_start(dense_grid, empty_grid, empty_layout) == 9


def test_sparse_trainer_loss_matches_manual_selected_patch_mse() -> None:
    trainer = object.__new__(Trainer)
    trainer.patch_size = (1, 2, 2)
    trainer.train_scheduler_latent = _StubScheduler()

    batch, channels, frames, height, width = 2, 2, 2, 4, 4
    target = torch.arange(
        batch * channels * frames * height * width, dtype=torch.float32
    ).reshape(batch, channels, frames, height, width)
    local = torch.tensor(
        [
            [[0, 3], [1, -1]],
            [[2, -1], [0, 3]],
        ]
    )
    valid = local >= 0
    timesteps = torch.tensor([[1.0, 3.0], [2.0, 5.0]])

    patch_volume = 4
    predictions = torch.empty(batch, frames * 2, patch_volume, channels)
    frame_error_sum = torch.zeros(batch, frames)
    frame_patch_count = torch.zeros(batch, frames)
    for batch_index in range(batch):
        for frame_index in range(frames):
            for slot in range(2):
                token_index = frame_index * 2 + slot
                if valid[batch_index, frame_index, slot]:
                    spatial_index = int(local[batch_index, frame_index, slot])
                    row, column = divmod(spatial_index, 2)
                    target_patch = (
                        target[
                            batch_index,
                            :,
                            frame_index,
                            row * 2 : (row + 1) * 2,
                            column * 2 : (column + 1) * 2,
                        ]
                        .permute(1, 2, 0)
                        .reshape(patch_volume, channels)
                    )
                    offset = float(1 + batch_index + token_index)
                    predictions[batch_index, token_index] = target_patch + offset
                    frame_error_sum[batch_index, frame_index] += offset**2
                    frame_patch_count[batch_index, frame_index] += 1
                else:
                    # A very wrong padded prediction must contribute exactly zero.
                    predictions[batch_index, token_index] = 10_000.0

    predictions.requires_grad_()
    input_dict = {
        "latent_dict": {
            "targets": target,
            "timesteps": timesteps,
            "rgb_motion_patch_indices": local,
            "rgb_motion_valid_mask": valid,
        }
    }
    actual = trainer._compute_latent_loss(
        input_dict, predictions.reshape(batch, frames * 2 * patch_volume, channels)
    )
    active = frame_patch_count > 0
    frame_error = frame_error_sum / frame_patch_count.clamp_min(1)
    expected = (frame_error * _StubScheduler.training_weight(timesteps))[active].mean()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert predictions.grad is not None
    assert not predictions.grad[~valid.reshape(batch, -1)].any()


def test_dense_model_and_trainer_paths_remain_legacy_equivalent() -> None:
    model = _tiny_model(sparse=False)
    latents = torch.randn(2, 2, 2, 4, 4)
    # Sparse-looking input must be ignored when the feature flag is disabled.
    ignored_motion_dict = {
        "noisy_latents": latents,
        "rgb_motion_patch_indices": torch.full((2, 2, 1), -1),
    }
    assert model._rgb_motion_layout(ignored_motion_dict, latents.shape) is None

    raw_dense = rearrange(
        latents,
        "b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)",
        pt=1,
        ph=2,
        pw=2,
    )
    expected_embedding = F.linear(
        raw_dense,
        model.patch_embedding_mlp.weight,
        model.patch_embedding_mlp.bias,
    )
    actual_embedding = model._input_embed(latents, input_type="latent")
    torch.testing.assert_close(actual_embedding, expected_embedding)

    trainer = object.__new__(Trainer)
    trainer.patch_size = (1, 2, 2)
    trainer.train_scheduler_latent = _StubScheduler()
    target = torch.randn(2, 2, 2, 4, 4)
    prediction_sequence = torch.randn(2, 2 * 2 * 2 * 4, 2)
    timesteps = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dense_input = {"latent_dict": {"targets": target, "timesteps": timesteps}}

    actual_loss = trainer._compute_latent_loss(dense_input, prediction_sequence)
    prediction_grid = data_seq_to_patch(
        trainer.patch_size,
        prediction_sequence,
        latent_num_frames=2,
        latent_height=4,
        latent_width=4,
        batch_size=2,
    )
    weights = _StubScheduler.training_weight(timesteps).reshape(2, 2)
    expected_loss = (
        (prediction_grid - target).square().mean(dim=(1, 3, 4)) * weights
    ).mean()
    torch.testing.assert_close(actual_loss, expected_loss)
