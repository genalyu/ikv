import builtins
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from n0_twam.preprocessing.dinov2 import (
    DinoPatchOutput,
    FrozenDinoV2PatchEncoder,
)


class FakeDino(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 2,
        prefix_tokens: int = 3,
        feature_dim: int = 4,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=patch_size)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.prefix_tokens = prefix_tokens
        self.feature_dim = feature_dim
        self.last_pixel_values = None

    def forward(self, *, pixel_values, return_dict=True):
        assert return_dict
        self.last_pixel_values = pixel_values.detach().clone()
        batch, _, height, width = pixel_values.shape
        patch_count = (height // self.config.patch_size) * (
            width // self.config.patch_size
        )
        prefix = torch.full(
            (batch, self.prefix_tokens, self.feature_dim),
            -100.0,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        patches = torch.arange(
            batch * patch_count * self.feature_dim,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).reshape(batch, patch_count, self.feature_dim)
        hidden = torch.cat((prefix, patches * self.scale), dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


def _encoder(model=None, **kwargs):
    return FrozenDinoV2PatchEncoder(
        model or FakeDino(),
        image_size=(4, 6),
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        **kwargs,
    )


def test_injected_model_is_frozen_and_permanently_eval() -> None:
    model = FakeDino()
    encoder = _encoder(model)

    assert not encoder.training
    assert not model.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    encoder.train(True)
    assert not encoder.training
    assert not model.training

    output = encoder(torch.zeros(2, 5, 7, 3, dtype=torch.uint8))
    assert not output.tokens.requires_grad
    assert output.tokens.grad_fn is None


def test_direct_resize_preserves_full_frame_and_never_crops() -> None:
    model = FakeDino()
    encoder = _encoder(model)
    rgb = torch.arange(1 * 2 * 4 * 3, dtype=torch.uint8).reshape(1, 2, 4, 3)

    encoder(rgb)

    expected = F.interpolate(
        rgb.permute(0, 3, 1, 2).float() / 255.0,
        size=(4, 6),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    assert model.last_pixel_values is not None
    torch.testing.assert_close(model.last_pixel_values, expected)


def test_cls_and_register_prefixes_are_skipped_by_patch_count() -> None:
    model = FakeDino(prefix_tokens=5, feature_dim=3)
    encoder = _encoder(model)
    output = encoder(torch.zeros(2, 3, 4, 6))

    assert isinstance(output, DinoPatchOutput)
    assert output.grid_size == (2, 3)
    assert output.resized_size == (4, 6)
    assert output.tokens.shape == (2, 2, 3, 3)
    expected = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 2, 3, 3)
    torch.testing.assert_close(output.tokens, expected)
    torch.testing.assert_close(output.flat_tokens, expected.flatten(1, 2))
    assert not torch.any(output.tokens == -100)


def test_channel_first_and_channel_last_inputs_are_equivalent() -> None:
    rgb_channels_last = torch.rand(2, 5, 7, 3)
    encoder = _encoder()
    channels_last = encoder(rgb_channels_last).tokens
    channels_first = encoder(rgb_channels_last.permute(0, 3, 1, 2)).tokens
    torch.testing.assert_close(channels_first, channels_last)


def test_float_0_255_range_is_explicit_and_normalized() -> None:
    model = FakeDino()
    encoder = _encoder(model, float_input_range="0_255")
    encoder(torch.full((1, 4, 6, 3), 255.0))
    torch.testing.assert_close(model.last_pixel_values, torch.ones(1, 3, 4, 6))

    default_encoder = _encoder()
    with pytest.raises(ValueError, match="configured as '0_1'"):
        default_encoder(torch.full((1, 4, 6, 3), 2.0))


def test_model_injection_does_not_import_transformers(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("injected-model path imported transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    output = _encoder()(torch.zeros(1, 4, 6, 3))
    assert output.tokens.shape == (1, 2, 3, 4)


def test_from_pretrained_is_local_only_and_uses_processor_statistics(
    monkeypatch,
) -> None:
    calls = []
    model = FakeDino()

    class FakeAutoImageProcessor:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(("processor", source, kwargs))
            return SimpleNamespace(
                image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6]
            )

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append(("model", source, kwargs))
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
    fake_transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    encoder = FrozenDinoV2PatchEncoder.from_pretrained(
        "local/dinov2",
        revision="fixed-revision",
        image_size=(4, 6),
    )

    assert [call[0] for call in calls] == ["processor", "model"]
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["revision"] == "fixed-revision" for call in calls)
    assert calls[1][2]["trust_remote_code"] is False
    assert calls[1][2]["torch_dtype"] is torch.float32
    torch.testing.assert_close(
        encoder.image_mean.flatten(), torch.tensor([0.1, 0.2, 0.3])
    )
    torch.testing.assert_close(
        encoder.image_std.flatten(), torch.tensor([0.4, 0.5, 0.6])
    )
    assert encoder.pretrained_source == "local/dinov2"
    assert encoder.pretrained_revision == "fixed-revision"


def test_local_only_load_failure_has_actionable_error(monkeypatch) -> None:
    class MissingProcessor:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            raise OSError("not cached")

    class UnusedModel:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            raise AssertionError("model load should not follow processor failure")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = MissingProcessor
    fake_transformers.AutoModel = UnusedModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="No network request was attempted") as exc:
        FrozenDinoV2PatchEncoder.from_pretrained("not-in-cache")
    assert "local_files_only=False" in str(exc.value)


@pytest.mark.parametrize(
    ("rgb", "error"),
    [
        (torch.zeros(1, 4, 6), "shape"),
        (torch.zeros(1, 3, 4, 3), "ambiguous"),
        (torch.zeros(1, 4, 6, 3, dtype=torch.int16), "uint8"),
        (torch.full((1, 4, 6, 3), float("nan")), "finite"),
    ],
)
def test_rgb_input_validation_is_clear(rgb, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _encoder()(rgb)


def test_patch_and_output_shape_validation_is_clear() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        FrozenDinoV2PatchEncoder(FakeDino(patch_size=3), image_size=(4, 6))

    class MissingPatchConfig(nn.Module):
        def forward(self, **kwargs):
            raise AssertionError

    with pytest.raises(ValueError, match="config.patch_size is missing"):
        FrozenDinoV2PatchEncoder(MissingPatchConfig())

    class TooShortDino(FakeDino):
        def forward(self, **kwargs):
            batch = kwargs["pixel_values"].shape[0]
            return {"last_hidden_state": torch.zeros(batch, 5, 3)}

    encoder = _encoder(TooShortDino())
    with pytest.raises(ValueError, match="at least 6 patch tokens"):
        encoder(torch.zeros(1, 4, 6, 3))


def test_output_dtype_can_be_preserved_or_selected() -> None:
    model = FakeDino().to(dtype=torch.float64)
    rgb = torch.zeros(1, 4, 6, 3)
    preserved = _encoder(model, output_dtype=None)(rgb)
    assert preserved.tokens.dtype is torch.float64

    selected = _encoder(FakeDino(), output_dtype=torch.float16)(rgb)
    assert selected.tokens.dtype is torch.float16
