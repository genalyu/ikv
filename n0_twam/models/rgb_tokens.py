"""RGB patch geometry, DINO pooling and independent token-index metadata."""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

def _positive_grid_size(size: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple((int(value) for value in size))
    if not result or any((value <= 0 for value in result)):
        raise ValueError(f'{name} must contain positive dimensions, got {result}.')
    return result

def _dino_to_channels_first(tokens: torch.Tensor, grid_size: Sequence[int] | None, name: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """Accept ``(B,H,W,D)`` or flattened ``(B,N,D)`` DINO patch tokens."""
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if not torch.is_floating_point(tokens):
        raise TypeError(f'{name} must be floating point, got {tokens.dtype}.')
    if tokens.ndim == 4:
        if grid_size is not None and tuple(tokens.shape[1:3]) != tuple(grid_size):
            raise ValueError(f'{name} grid is {tuple(tokens.shape[1:3])}, but grid_size is {tuple(grid_size)}.')
        if tokens.shape[-1] < 1:
            raise ValueError(f'{name} feature dimension cannot be empty.')
        height, width = (int(tokens.shape[1]), int(tokens.shape[2]))
        return (tokens.permute(0, 3, 1, 2).contiguous(), (height, width))
    if tokens.ndim == 3:
        if grid_size is None:
            raise ValueError(f'grid_size is required when {name} has flattened shape (B,N,D).')
        normalized_grid = _positive_grid_size(grid_size, 'grid_size')
        if len(normalized_grid) != 2:
            raise ValueError(f'grid_size must have two dimensions, got {normalized_grid}.')
        height, width = normalized_grid
        if tokens.shape[1] != height * width:
            raise ValueError(f'{name} has {tokens.shape[1]} tokens, expected {height * width} for grid {(height, width)}.')
        if tokens.shape[-1] < 1:
            raise ValueError(f'{name} feature dimension cannot be empty.')
        mapped = tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1])
        return (mapped.permute(0, 3, 1, 2).contiguous(), (height, width))
    raise ValueError(f'{name} must have shape (B,H,W,D) or (B,N,D), got {tuple(tokens.shape)}.')

def pool_dino_to_grid(dino_tokens: torch.Tensor, target_size: Sequence[int], *, source_grid_size: Sequence[int] | None=None) -> torch.Tensor:
    """Pool DINO tokens to a new grid and return channels-last features.

    Average pooling is used when reducing 16x16 DINO features to the usual 8x8
    WAN-transformer grid. Bilinear interpolation handles non-integer or larger
    target grids.
    """
    mapped, _ = _dino_to_channels_first(dino_tokens, source_grid_size, 'dino_tokens')
    target = _positive_grid_size(target_size, 'target_size')
    if len(target) != 2:
        raise ValueError(f'target_size must have two dimensions, got {target}.')
    source = tuple(mapped.shape[-2:])
    if source != target:
        if target[0] <= source[0] and target[1] <= source[1]:
            mapped = F.adaptive_avg_pool2d(mapped, target)
        else:
            mapped = F.interpolate(mapped, size=target, mode='bilinear', align_corners=False)
    return mapped.permute(0, 2, 3, 1).contiguous()

@dataclass(frozen=True)
class TokenIndex:
    """Independent semantic index carried beside, never inside, content values.

    Formal index fields are ``{world_time_id, dino, neoforce,
    observation_flag}``, where ``observation_flag=1`` means observed and ``0``
    means predicted. ``visual_valid`` and ``tactile_valid`` are presence masks so
    a padded zero feature cannot be confused with a real feature. They are support
    metadata, not extra index dimensions.
    """
    world_time_id: torch.Tensor
    dino: torch.Tensor
    neoforce: torch.Tensor
    observation_flag: torch.Tensor
    visual_valid: torch.Tensor
    tactile_valid: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {'world_time_id': self.world_time_id, 'dino': self.dino, 'neoforce': self.neoforce, 'observation_flag': self.observation_flag, 'visual_valid': self.visual_valid, 'tactile_valid': self.tactile_valid}
        if not all((isinstance(value, torch.Tensor) for value in tensors.values())):
            raise TypeError('Every TokenIndex field must be a torch.Tensor.')
        if self.world_time_id.ndim != 2:
            raise ValueError('world_time_id must have shape (B,K).')
        if self.world_time_id.dtype == torch.bool or self.world_time_id.is_complex():
            raise TypeError('world_time_id must contain real numeric time values.')
        if self.world_time_id.dtype.is_floating_point and (not torch.isfinite(self.world_time_id).all()):
            raise ValueError('world_time_id must contain only finite values.')
        shape = self.world_time_id.shape
        if self.dino.ndim != 3 or self.dino.shape[:2] != shape:
            raise ValueError('dino must have shape (B,K,D_dino).')
        if self.neoforce.ndim != 3 or self.neoforce.shape[:2] != shape:
            raise ValueError('neoforce must have shape (B,K,D_neoforce).')
        dino_width = self.dino.shape[-1]
        neoforce_width = self.neoforce.shape[-1]
        if dino_width == 0 and neoforce_width == 0:
            raise ValueError('DINO and NeoForce feature dimensions cannot both be zero.')
        if not torch.is_floating_point(self.dino) or not torch.is_floating_point(self.neoforce):
            raise TypeError('dino and neoforce must be floating point tensors.')
        for name, value in (('observation_flag', self.observation_flag), ('visual_valid', self.visual_valid), ('tactile_valid', self.tactile_valid)):
            if value.shape != shape:
                raise ValueError(f'{name} must have shape {tuple(shape)}.')
        if self.visual_valid.dtype != torch.bool or self.tactile_valid.dtype != torch.bool:
            raise TypeError('visual_valid and tactile_valid must have dtype bool.')
        if dino_width == 0 and torch.any(self.visual_valid):
            raise ValueError('visual_valid must be all False when the DINO feature width is zero.')
        if neoforce_width == 0 and torch.any(self.tactile_valid):
            raise ValueError('tactile_valid must be all False when the NeoForce feature width is zero.')
        if self.observation_flag.dtype == torch.bool:
            pass
        elif self.observation_flag.dtype.is_floating_point or self.observation_flag.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if torch.any((self.observation_flag != 0) & (self.observation_flag != 1)):
                raise ValueError('observation_flag values must be exactly 0 or 1.')
        else:
            raise TypeError('observation_flag must be bool, integer, or floating point.')
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError('All TokenIndex fields must be on the same device.')

    @property
    def observed(self) -> torch.Tensor:
        return self.observation_flag.to(torch.bool)

    @property
    def predicted(self) -> torch.Tensor:
        return ~self.observed

    def validate_presence(self, token_valid_mask: torch.Tensor | None=None) -> None:
        """Require DINO or NeoForce for every real (non-padding) token."""
        present = self.visual_valid | self.tactile_valid
        if token_valid_mask is None:
            token_valid_mask = torch.ones_like(present)
        elif token_valid_mask.shape != present.shape or token_valid_mask.dtype != torch.bool:
            raise ValueError('token_valid_mask must be bool and match the TokenIndex (B,K) shape.')
        if token_valid_mask.device != present.device:
            raise ValueError('token_valid_mask and TokenIndex must share a device.')
        if torch.any(token_valid_mask & ~present):
            raise ValueError('Every valid token must contain at least one of DINO or NeoForce.')
__all__ = ['EgoMotionCompensatedMotionDetector', 'MotionDetectionResult', 'SparsePatchBatch', 'SparsePatchGather', 'SparsePatchScatter', 'TokenIndex', 'gather_grid_features', 'motion_result_to_sidecar', 'padded_indices_from_mask', 'patchify_latents', 'pool_dino_to_grid', 'unpatchify_latents']
