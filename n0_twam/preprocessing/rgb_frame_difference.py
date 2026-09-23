"""Pure RGB adjacent-frame selection; no geometry or learned encoder required."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F


def rgb_tensor(value):
    x = torch.as_tensor(value)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or (x.shape[1] == 3) == (x.shape[-1] == 3):
        raise ValueError('RGB must unambiguously be [T,3,H,W] or [T,H,W,3]')
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    if x.dtype == torch.uint8:
        x = x.float() / 255
    elif not x.is_floating_point():
        raise ValueError('RGB must be uint8 or float in [0,1]')
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('RGB must be finite in [0,1]')
    return x.float()


def parse_rgb_cameras(payload, camera_keys):
    if tuple(payload.get('camera_keys', ())) != tuple(camera_keys):
        raise ValueError('camera_keys must match configured camera order')
    cameras = payload.get('cameras', {})
    if set(cameras) != set(camera_keys) or not camera_keys:
        raise ValueError('camera set mismatch')
    result = {}
    for key in camera_keys:
        raw = torch.as_tensor(cameras[key]['rgb'])
        if raw.ndim == 3:
            raw = raw.unsqueeze(0)
        rgb_tensor(raw)  # validate without changing observation binding representation
        result[key] = SimpleNamespace(rgb=raw, num_frames=raw.shape[0])
    counts = {v.num_frames for v in result.values()}
    if len(counts) != 1:
        raise ValueError('camera frame counts must match')
    return result, counts.pop()


class RGBFrameDifferencePreprocessor:
    def __init__(self, *, camera_keys, height, width, patch_size=(1,2,2), threshold=0.02, dino_encoder=None):
        self.dino_encoder = dino_encoder
        self.camera_keys = tuple(camera_keys)
        self.height, self.width = int(height), int(width)
        self.patch_size = tuple(patch_size)
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be in [0,1]')
        if self.patch_size[0] != 1 or height % (16*patch_size[1]) or width % (16*patch_size[2]):
            raise ValueError('image size must align with WAN spatial patches; temporal patch=1')

    def __call__(self, cameras, *, anchor_indices, world_time_ids,
                 previous_frames=None, observation_flag=1):
        if tuple(cameras) != self.camera_keys:
            raise ValueError('camera order mismatch')
        anchors = torch.as_tensor(anchor_indices, dtype=torch.long).tolist()
        times = torch.as_tensor(world_time_ids, dtype=torch.long)
        count = next(iter(cameras.values())).num_frames
        if not anchors or anchors != sorted(set(anchors)) or anchors[0] < 0 or anchors[-1] != count-1:
            raise ValueError('anchors must increase and end at final raw frame')
        if times.numel() != len(anchors) or (times < 0).any():
            raise ValueError('world time mismatch')
        if previous_frames is not None and set(previous_frames) != set(self.camera_keys):
            raise ValueError('previous camera set mismatch')
        gh = self.height // (16*self.patch_size[1])
        gw = self.width // (16*self.patch_size[2])
        per_camera = []
        features = []
        for key in self.camera_keys:
            x = rgb_tensor(cameras[key].rgb)
            if len(x) != count:
                raise ValueError('unaligned camera frames')
            x = F.interpolate(x, (self.height,self.width), mode='bilinear', align_corners=False)
            previous = None
            if previous_frames is not None:
                item = previous_frames[key]
                previous = rgb_tensor(item['rgb'] if isinstance(item,dict) else item.rgb)[-1:]
                previous = F.interpolate(previous.to(x), (self.height,self.width), mode='bilinear', align_corners=False)
            ref = torch.cat((x[:1] if previous is None else previous, x[:-1]), 0)
            scores = F.adaptive_avg_pool2d((x-ref).abs().mean(1,keepdim=True), (gh,gw))[:,0]
            if previous is None:
                scores[0] = 1.0  # full initial frame, never truncated to max_tokens
            start = 0
            selected = []
            for end in anchors:
                selected.append(scores[start:end+1].amax(0))
                start = end+1
            per_camera.append(torch.stack(selected))
            if self.dino_encoder is not None:
                with torch.no_grad():
                    encoded = self.dino_encoder(x[anchors])
                tokens = encoded if isinstance(encoded, torch.Tensor) else encoded.tokens
                if tokens.ndim == 3:
                    grid = encoded.grid_size
                    tokens = tokens.reshape(len(anchors), *grid, -1)
                grid_features = F.interpolate(tokens.permute(0,3,1,2).float(),
                    (gh,gw), mode='bilinear', align_corners=False).permute(0,2,3,1)
                features.append(grid_features.cpu())
        scores = torch.cat(per_camera, dim=2).cpu()
        valid = (scores > self.threshold).flatten(1)
        if previous_frames is None:
            valid[0] = True
        n, capacity = valid.shape
        dino = (torch.cat(features,dim=2).reshape(n,capacity,-1)
                if features else torch.zeros(n,capacity,1))
        dino = dino.masked_fill(~valid[...,None],0)
        indices = torch.arange(capacity).expand(n,-1).clone().masked_fill(~valid,-1)
        # Full capacity padding preserves every changed patch and a dense first frame.
        return dict(motion_indices=indices, motion_valid_mask=valid,
            motion_scores=scores.flatten(1).masked_fill(~valid,0),
            world_time_id=times[:,None].expand_as(indices).clone().masked_fill(~valid,-1),
            dino_features=dino, neoforce_features=torch.zeros(n,capacity,0),
            observation_flag=torch.full_like(indices,int(observation_flag)).masked_fill(~valid,0),
            visual_valid=valid.clone() if features else torch.zeros_like(valid), tactile_valid=torch.zeros_like(valid),
            camera_keys=list(self.camera_keys), patch_size=self.patch_size,
            spatial_grid_shape=(gh,gw*len(self.camera_keys)), latent_num_frames=n,
            provenance={'producer':'RGBFrameDifferencePreprocessor','anchor_indices':anchors,
                        'threshold':self.threshold,'first_frame_policy':'all'})
