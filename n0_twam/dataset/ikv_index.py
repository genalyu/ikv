"""Optional dense index sidecar. Content latents and actions remain unchanged."""
from pathlib import Path
import torch

def load_dense_index(path, *, camera_keys, patch_size, grid_shape,
                     latent_frame_ids, full_frames, start=None, end=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = dict(camera_keys=list(camera_keys), patch_size=list(patch_size),
                    spatial_grid_shape=list(grid_shape), frame_ids=list(latent_frame_ids))
    for name, value in expected.items():
        supplied = payload.get(name)
        if supplied is None or list(supplied) != value:
            raise ValueError(f"dense IKV index {name} differs from encoded video")
    spatial = grid_shape[0]*grid_shape[1]
    result = {}
    for name in ("dino_features", "neoforce_features"):
        value = payload.get(name)
        if value is None:
            value = torch.empty(full_frames, spatial, 0)
        value = torch.as_tensor(value)
        if value.ndim != 3 or tuple(value.shape[:2]) != (full_frames, spatial) or not torch.isfinite(value).all():
            raise ValueError(f"dense IKV {name} requires finite [F,spatial,D]")
        result[name] = value[slice(start,end)].float()
    return result
