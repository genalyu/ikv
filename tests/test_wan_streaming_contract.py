from __future__ import annotations

import torch
from diffusers import AutoencoderKLWan

from n0_twam.models.utils import WanVAEStreamingWrapper


def _build_tiny_wan_vae() -> AutoencoderKLWan:
    """Build the real diffusers WAN encoder with a deliberately tiny width."""
    torch.manual_seed(1234)
    return AutoencoderKLWan(
        base_dim=2,
        decoder_base_dim=2,
        z_dim=2,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
        in_channels=3,
        out_channels=3,
        patch_size=None,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
    ).eval()


def test_real_wan_streaming_encoder_preserves_causal_four_frame_chunks() -> None:
    """Lock down the cold-1 / warm-8 contract used by the online server."""
    vae = _build_tiny_wan_vae()
    streaming_vae = WanVAEStreamingWrapper(vae)

    inputs = torch.Generator().manual_seed(5678)
    cold_seed = torch.randn(1, 3, 1, 8, 8, generator=inputs)
    warm_frames = torch.randn(1, 3, 8, 8, 8, generator=inputs)
    changed_warm_frames = warm_frames.clone()
    changed_warm_frames[:, :, 4:8] *= -1

    def encode_sequence(warm_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        streaming_vae.clear_cache()
        with torch.no_grad():
            cold_latent = streaming_vae.encode_chunk(cold_seed)
            warm_latents = streaming_vae.encode_chunk(warm_chunk)
        return cold_latent, warm_latents

    cold_a, warm_a = encode_sequence(warm_frames)
    cold_b, warm_b = encode_sequence(changed_warm_frames)

    # A fresh one-frame seed creates one latent. With that cache warm, eight
    # further raw frames create two causal four-frame latents.
    assert cold_a.shape == (1, 4, 1, 1, 1)
    assert warm_a.shape == (1, 4, 2, 1, 1)
    assert cold_b.shape == cold_a.shape
    assert warm_b.shape == warm_a.shape

    # Resetting and replaying the same seed must reproduce the cold state. The
    # first warm latent only sees warm frames 0..3, while the second responds to
    # the changed warm frames 4..7.
    torch.testing.assert_close(cold_a, cold_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(warm_a[:, :, 0], warm_b[:, :, 0], rtol=1e-6, atol=1e-7)
    second_latent_delta = (warm_a[:, :, 1] - warm_b[:, :, 1]).abs().max()
    assert second_latent_delta.item() > 1e-5
