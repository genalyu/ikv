"""Frozen DINOv2 patch feature extraction without implicit network access.

The RGB-motion path treats DINO features as semantic index metadata.  This
module therefore remains separate from the N0-TWAM transformer and never adds
the extracted features to WAN content embeddings or Q/K/V values.

Construct :class:`FrozenDinoV2PatchEncoder` with an existing model for tests or
for applications which manage model loading themselves.  The optional
``from_pretrained`` constructor imports ``transformers`` lazily and defaults to
``local_files_only=True``; importing this module never loads weights or touches
the network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


FloatInputRange = Literal["0_1", "0_255"]


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(int(item) for item in value)
        if len(result) != 2:
            raise ValueError(f"{name} must contain two values, got {result}.")
    else:
        raise TypeError(f"{name} must be an int or a two-element sequence.")
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive, got {result}.")
    return result


def _triplet(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a three-element sequence.")
    result = tuple(float(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain three values, got {result}.")
    if not all(torch.isfinite(torch.tensor(item)) for item in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _model_patch_size(model: nn.Module) -> int | Sequence[int] | None:
    config = getattr(model, "config", None)
    return getattr(config, "patch_size", None) if config is not None else None


def _model_device_and_dtype(
    model: nn.Module, fallback_device: torch.device
) -> tuple[torch.device, torch.dtype]:
    for value in (*model.parameters(), *model.buffers()):
        if torch.is_floating_point(value):
            return value.device, value.dtype
    return fallback_device, torch.float32


@dataclass(frozen=True)
class DinoPatchOutput:
    """Dense, channels-last DINO patch tokens for one image batch."""

    tokens: torch.Tensor
    grid_size: tuple[int, int]
    resized_size: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, torch.Tensor) or self.tokens.ndim != 4:
            shape = (
                tuple(self.tokens.shape)
                if isinstance(self.tokens, torch.Tensor)
                else None
            )
            raise ValueError(
                "tokens must be a tensor with shape (B,H_patch,W_patch,D), "
                f"got {shape}."
            )
        if tuple(self.tokens.shape[1:3]) != tuple(self.grid_size):
            raise ValueError(
                f"token grid {tuple(self.tokens.shape[1:3])} does not match "
                f"grid_size {self.grid_size}."
            )
        if self.tokens.shape[-1] <= 0:
            raise ValueError("DINO feature width must be positive.")
        if not torch.is_floating_point(self.tokens):
            raise TypeError("DINO patch tokens must be floating point.")

    @property
    def flat_tokens(self) -> torch.Tensor:
        """Return the same features as ``(B, H_patch * W_patch, D)``."""

        return self.tokens.flatten(1, 2)


class FrozenDinoV2PatchEncoder(nn.Module):
    """Run a permanently frozen DINO-style model and return patch tokens.

    Input is RGB in ``(B,3,H,W)`` or ``(B,H,W,3)`` form. ``uint8`` inputs use
    the range 0..255. Floating-point input range is explicit through
    ``float_input_range`` and defaults to 0..1, avoiding value-range guessing.

    Images are resized directly to ``image_size`` with no crop. This property is
    important for the RGB-D motion detector: the DINO grid continues to cover
    the same complete field of view as depth and camera intrinsics.

    The wrapped model must follow the Hugging Face DINO convention and return a
    ``last_hidden_state`` tensor of shape ``(B,L,D)`` (a mapping or a tuple whose
    first item is that tensor is also accepted). Patch tokens are taken from the
    final ``H_patch * W_patch`` positions, safely skipping CLS and any register
    tokens at the beginning of the sequence.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        patch_size: int | Sequence[int] | None = None,
        image_size: int | Sequence[int] = (224, 224),
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        float_input_range: FloatInputRange = "0_1",
        output_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")
        resolved_patch = patch_size
        if resolved_patch is None:
            resolved_patch = _model_patch_size(model)
        if resolved_patch is None:
            raise ValueError(
                "patch_size was not provided and model.config.patch_size is missing."
            )

        self.patch_size = _pair(resolved_patch, "patch_size")
        self.image_size = _pair(image_size, "image_size")
        if any(image % patch for image, patch in zip(self.image_size, self.patch_size)):
            raise ValueError(
                f"image_size {self.image_size} must be divisible by patch_size "
                f"{self.patch_size}."
            )
        mean = _triplet(image_mean, "image_mean")
        std = _triplet(image_std, "image_std")
        if any(item <= 0 for item in std):
            raise ValueError(f"image_std values must be positive, got {std}.")
        if float_input_range not in ("0_1", "0_255"):
            raise ValueError(
                "float_input_range must be either '0_1' or '0_255', got "
                f"{float_input_range!r}."
            )
        if (
            output_dtype is not None
            and not torch.empty((), dtype=output_dtype).is_floating_point()
        ):
            raise TypeError(
                "output_dtype must be a floating-point torch dtype or None."
            )

        self.model = model
        self.float_input_range = float_input_range
        self.output_dtype = output_dtype
        self.register_buffer(
            "image_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.model.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        local_files_only: bool = True,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        revision: str | None = None,
        image_size: int | Sequence[int] = (224, 224),
        patch_size: int | Sequence[int] | None = None,
        float_input_range: FloatInputRange = "0_1",
        output_dtype: torch.dtype | None = torch.float32,
    ) -> "FrozenDinoV2PatchEncoder":
        """Load weights and normalization metadata, locally by default.

        Set ``local_files_only=False`` explicitly to authorize Hugging Face
        downloads. No online fallback is attempted after a local-only failure.
        """

        try:
            from transformers import AutoImageProcessor, AutoModel
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Loading DINOv2 weights requires the 'transformers' package. "
                "Install the project dependencies, or inject an existing "
                "torch.nn.Module into FrozenDinoV2PatchEncoder(model=...)."
            ) from exc

        source = str(pretrained_model_name_or_path)
        load_kwargs: dict[str, Any] = {
            "local_files_only": bool(local_files_only),
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            processor = AutoImageProcessor.from_pretrained(source, **load_kwargs)
            model = AutoModel.from_pretrained(
                source,
                torch_dtype=torch_dtype,
                trust_remote_code=False,
                **load_kwargs,
            )
        except OSError as exc:
            if local_files_only:
                detail = (
                    "The model or image-processor files were not found at the "
                    "given local path or in the Hugging Face cache. No network "
                    "request was attempted. Provide a complete local checkpoint, "
                    "or explicitly pass local_files_only=False to permit download."
                )
            else:
                detail = (
                    "The model or image-processor files could not be loaded from "
                    "the requested source."
                )
            raise RuntimeError(
                f"Failed to load DINOv2 from {source!r}. {detail}"
            ) from exc

        mean = getattr(processor, "image_mean", None)
        std = getattr(processor, "image_std", None)
        if mean is None or std is None:
            raise ValueError(
                "The loaded image processor must define three-channel image_mean "
                "and image_std values."
            )
        model = model.to(device=torch.device(device), dtype=torch_dtype)
        encoder = cls(
            model,
            patch_size=patch_size,
            image_size=image_size,
            image_mean=mean,
            image_std=std,
            float_input_range=float_input_range,
            output_dtype=output_dtype,
        )
        encoder.pretrained_source = source
        encoder.pretrained_revision = revision
        return encoder

    def train(self, mode: bool = True) -> "FrozenDinoV2PatchEncoder":
        """Keep the wrapper and its model in eval mode permanently."""

        super().train(False)
        self.model.eval()
        return self

    def _prepare_pixel_values(self, rgb: torch.Tensor) -> torch.Tensor:
        if not isinstance(rgb, torch.Tensor) or rgb.ndim != 4:
            shape = tuple(rgb.shape) if isinstance(rgb, torch.Tensor) else None
            raise ValueError(
                "rgb must be a tensor with shape (B,3,H,W) or (B,H,W,3), "
                f"got {shape}."
            )
        channels_first = rgb.shape[1] == 3
        channels_last = rgb.shape[-1] == 3
        if channels_first == channels_last:
            raise ValueError(
                "rgb channel dimension is ambiguous or missing; exactly one of "
                "dimension 1 and the final dimension must have size 3."
            )
        if channels_last:
            rgb = rgb.permute(0, 3, 1, 2)
        if rgb.shape[0] <= 0 or rgb.shape[2] <= 0 or rgb.shape[3] <= 0:
            raise ValueError("rgb batch and spatial dimensions must be non-empty.")

        if rgb.dtype == torch.uint8:
            values = rgb.to(torch.float32).div_(255.0)
        elif torch.is_floating_point(rgb):
            values = rgb.to(torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError("rgb must contain only finite values.")
            upper = 1.0 if self.float_input_range == "0_1" else 255.0
            if torch.any(values < 0) or torch.any(values > upper):
                raise ValueError(
                    f"floating rgb configured as {self.float_input_range!r} must "
                    f"lie in [0, {upper:g}]."
                )
            if self.float_input_range == "0_255":
                values = values / 255.0
        else:
            raise TypeError(
                "rgb must use uint8 or a floating-point dtype; integer RGB "
                "types other than uint8 are ambiguous."
            )

        # Exact full-frame resize: deliberately no shortest-edge resize and no
        # center crop, so normalized patch locations stay aligned with depth.
        values = F.interpolate(
            values,
            size=self.image_size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        values = (values - self.image_mean.to(values.device)) / self.image_std.to(
            values.device
        )
        model_device, model_dtype = _model_device_and_dtype(self.model, values.device)
        return values.to(device=model_device, dtype=model_dtype)

    @staticmethod
    def _last_hidden_state(output: Any) -> torch.Tensor:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, Mapping):
            hidden = output.get("last_hidden_state")
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(
                "DINO model output must expose a tensor named last_hidden_state, "
                "or return it as the first tuple item."
            )
        return hidden

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> DinoPatchOutput:
        pixel_values = self._prepare_pixel_values(rgb)
        output = self.model(pixel_values=pixel_values, return_dict=True)
        hidden = self._last_hidden_state(output)
        grid = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        patch_count = grid[0] * grid[1]
        if hidden.ndim != 3:
            raise ValueError(
                "DINO last_hidden_state must have shape (B,L,D), got "
                f"{tuple(hidden.shape)}."
            )
        if hidden.shape[0] != rgb.shape[0]:
            raise ValueError(
                f"DINO output batch is {hidden.shape[0]}, expected {rgb.shape[0]}."
            )
        if hidden.shape[1] < patch_count:
            raise ValueError(
                f"DINO output has {hidden.shape[1]} sequence tokens, but image/"
                f"patch sizes require at least {patch_count} patch tokens."
            )
        if hidden.shape[2] <= 0 or not torch.is_floating_point(hidden):
            raise TypeError(
                "DINO hidden states must have a positive floating feature width."
            )

        # DINO-style special tokens are prefixes. Taking the final patch_count
        # entries handles both plain CLS and CLS + register-token variants.
        patches = hidden[:, -patch_count:, :].reshape(
            hidden.shape[0], grid[0], grid[1], hidden.shape[2]
        )
        if not torch.isfinite(patches).all():
            raise ValueError("DINO patch tokens contain non-finite values.")
        if self.output_dtype is not None:
            patches = patches.to(dtype=self.output_dtype)
        return DinoPatchOutput(
            tokens=patches.detach().contiguous(),
            grid_size=grid,
            resized_size=self.image_size,
        )


__all__ = ["DinoPatchOutput", "FrozenDinoV2PatchEncoder"]
