"""Frozen official NeoForce features; kept separate from WAN content embeddings.

Input force/mask must come from a calibrated conversion or a validated physical
sensor adapter. Do not apply the real sensor's fixed homography to NeoSim RGB.
Returned patch_t is in the tactile grid, NOT the visual camera grid.
"""
from pathlib import Path
import torch
import torch.nn.functional as F


class FrozenNeoForceEncoder:
    def __init__(self, model, force_scale):
        self.model = model.eval().requires_grad_(False)
        self.scale = torch.as_tensor(force_scale, dtype=torch.float32)
        if self.scale.shape != (6,) or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError('NeoForce requires six finite positive training force scales')

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        # Install/pin the upstream neoforce package separately. No implicit download.
        from neoforce.models import NeoForceConfig, NeoForceEncoder
        ck = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        cfg = NeoForceConfig(**{k: v for k, v in ck['model_cfg'].items()
                                if k in NeoForceConfig.__dataclass_fields__})
        # Loading an index-only encoder must not alter the policy's sampling RNG.
        with torch.random.fork_rng(devices=[]):
            model = NeoForceEncoder(cfg)
        result = model.load_state_dict(ck['model'], strict=False)
        if result.missing_keys or any(not key.startswith('jepa_predictor.') for key in result.unexpected_keys):
            raise ValueError(f'NeoForce checkpoint mismatch: {result}')
        norm = ck.get('force_norm') or {}
        data = ck.get('data_cfg') or {}
        scale = norm.get('scale', data.get('force_std'))
        mean = norm.get('mean', data.get('force_mean'))
        if scale is None or (mean is not None and any(float(x) != 0 for x in mean)):
            raise ValueError('Missing training force scale or non-zero normalization mean')
        return cls(model.to(device=device, dtype=torch.float32), scale)

    @torch.inference_mode()
    def __call__(self, rgb, force, mask):
        """[B,T,3,H,W] RGB [0,1]; [B,T,6,Hf,Wf] force and binary mask.

        The caller supplies a causal clip only. A raw nonzero latent is never
        used as evidence of contact; response is calculated from the masked
        physical input before normalization, separately for each sensor.
        """
        device = next(self.model.parameters()).device
        rgb, force, mask = [torch.as_tensor(x, device=device).float() for x in (rgb, force, mask)]
        if rgb.ndim != 5 or rgb.shape[2] != 3 or force.ndim != 5 or force.shape[2] != 6:
            raise ValueError('RGB/force must be [B,T,3,H,W]/[B,T,6,Hf,Wf]')
        if rgb.shape[:2] != force.shape[:2] or mask.shape != force.shape:
            raise ValueError('NeoForce RGB/force/mask frame alignment mismatch')
        if any(not torch.isfinite(x).all() for x in (rgb, force, mask)):
            raise ValueError('NeoForce inputs must be finite')
        if (rgb < 0).any() or (rgb > 1).any() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError('RGB must be [0,1] and contact mask must be binary')
        b, t = rgb.shape[:2]
        if t < 1 or t > self.model.cfg.temporal_max_frames:
            raise ValueError('NeoForce clip exceeds temporal position capacity')
        size = tuple(self.model.cfg.image_size)
        rgb = F.interpolate(rgb.flatten(0, 1), size, mode='bilinear', align_corners=False).reshape(b, t, 3, *size)
        mean = rgb.new_tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
        std = rgb.new_tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
        physical = force * mask
        normalized = physical / self.scale.to(device).view(1, 1, 6, 1, 1)
        out = self.model((rgb - mean) / std, force=normalized, mask=mask,
                         touch_available=torch.ones(b, t, device=device, dtype=torch.bool))
        gh, gw = (s // self.model.cfg.patch_size for s in size)
        tokens = out['patch_t'].float().reshape(b, t, gh, gw, -1)
        if not torch.isfinite(tokens).all():
            raise ValueError('NeoForce emitted nonfinite patch tokens')
        response = physical.reshape(b, t, 2, 3, *physical.shape[-2:]).abs().amax(3)
        result = {'tokens': tokens.detach(), 'response': response.detach()}
        if 'patch_v' in out:
            result['visual_tokens'] = out['patch_v'].float().reshape(b, t, gh, gw, -1).detach()
        return result


@torch.no_grad()
def pool_contact_tokens(encoded, target_size):
    """Sensor-local metadata [B,2,T,H,W,D], never visual correspondences.

    The official encoder fuses the sensor pair into one patch_t grid. Both
    sensor records use that contextual feature, but each is gated by its OWN
    contact response. Pool response before thresholding so a small real contact
    is not discarded. This does not infer where the contact is in a RGB image.
    """
    tokens, response = encoded['tokens'], encoded['response']
    if tokens.ndim != 5 or response.ndim != 5 or response.shape[2] != 2:
        raise ValueError('tokens/response must be [B,T,H,W,D]/[B,T,2,Hf,Wf]')
    if tokens.shape[:2] != response.shape[:2] or tokens.device != response.device:
        raise ValueError('token and response clips/devices must match')
    if (not torch.isfinite(tokens).all() or not torch.isfinite(response).all()
            or (response < 0).any()):
        raise ValueError('tokens and nonnegative response must be finite')
    if len(target_size) != 2 or any(type(x) is not int or x <= 0 for x in target_size):
        raise ValueError('target_size must contain two positive integers')
    b, t, _, _, d = tokens.shape
    pooled = F.adaptive_avg_pool2d(tokens.permute(0, 1, 4, 2, 3).flatten(0, 1), target_size)
    pooled = pooled.reshape(b, t, d, *target_size).permute(0, 1, 3, 4, 2)
    amplitude = F.adaptive_max_pool2d(response.flatten(0, 2).unsqueeze(1), target_size)
    amplitude = amplitude.reshape(b, t, 2, *target_size).permute(0, 2, 1, 3, 4)
    features = pooled[:, None].expand(-1, 2, -1, -1, -1, -1).clone()
    features[amplitude == 0] = 0
    return {'neoforce': features, 'response': amplitude}


@torch.no_grad()
def encode_causal_contact_frames(encoder, rgb, force, mask, anchors, target_size,
                                 context_frames=4):
    """Encode only observed endpoints, using no RGB/force later than each anchor.

    Clips are [B,raw_T,C,H,W]. `anchors` must be supplied by the SAME streaming
    VAE timing convention used for the corresponding WAN tokens. This helper
    does not guess timestamps, resample frame rate, extrapolate missing history,
    or turn one observation into several denoising-time contact measurements.
    A short beginning uses its available history instead of fake observations.
    Output ordering is [B,sensor,WAN_T,H,W,D], matching tactile stream layout.
    """
    if type(context_frames) is not int or context_frames < 1:
        raise ValueError('context_frames must be a positive integer')
    if any(not isinstance(x, torch.Tensor) or x.ndim != 5 for x in (rgb, force, mask)):
        raise ValueError('causal contact inputs must be rank-five tensors')
    if rgb.shape[:2] != force.shape[:2] or force.shape != mask.shape:
        raise ValueError('causal contact observation clips must align')
    ids = torch.as_tensor(anchors)
    if (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or ids.numel() == 0
            or (ids < 0).any() or (ids >= rgb.shape[1]).any()
            or (ids[1:] <= ids[:-1]).any()):
        raise ValueError('anchors must be nonempty strictly increasing in-range integer frame IDs')
    all_features, all_response = [], []
    for anchor in ids.tolist():
        start = max(0, anchor + 1 - context_frames)
        encoded = encoder(rgb[:, start:anchor + 1], force[:, start:anchor + 1],
                          mask[:, start:anchor + 1])
        # The clip's older frames provide context only, never duplicate KV rows.
        endpoint = {'tokens': encoded['tokens'][:, -1:],
                    'response': encoded['response'][:, -1:]}
        pooled = pool_contact_tokens(endpoint, target_size)
        all_features.append(pooled['neoforce'])
        all_response.append(pooled['response'])
    return {'neoforce': torch.cat(all_features, dim=2),
            'response': torch.cat(all_response, dim=2),
            'raw_frame_ids': ids.detach().clone()}
