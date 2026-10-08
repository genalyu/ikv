"""Optional dense index sidecar. Content latents and actions remain unchanged."""
from pathlib import Path
import torch

def load_dense_index(path, *, camera_keys, patch_size, grid_shape,
                     latent_frame_ids, full_frames, start=None, end=None, semantic_expected=None,
                     require_task=False):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected = dict(camera_keys=list(camera_keys), patch_size=list(patch_size),
                    spatial_grid_shape=list(grid_shape), frame_ids=list(latent_frame_ids))
    for name, value in expected.items():
        supplied = payload.get(name)
        if supplied is None or list(supplied) != value:
            raise ValueError(f"dense IKV index {name} differs from encoded video")
    if semantic_expected is not None and payload.get("semantic_provenance") != semantic_expected:
        raise ValueError("semantic cache backend/assets/prompt/preprocessing changed; regenerate sidecars")
    if require_task and "task_relevance" not in payload:
        raise ValueError("task scoring requires regenerated semantic sidecars")
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
    if "task_relevance" in payload:
        value = torch.as_tensor(payload["task_relevance"])
        if value.shape != (full_frames, spatial) or not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError("dense task_relevance requires finite [F,spatial] in [0,1]")
        result["task_relevance"] = value[slice(start, end)].float()
    frame_neo = payload.get("frame_neoforce_features")
    if frame_neo is not None:
        frame_neo = torch.as_tensor(frame_neo)
        if frame_neo.ndim != 2 or len(frame_neo) != full_frames or not torch.isfinite(frame_neo).all():
            raise ValueError("frame_neoforce_features requires finite [F,D], zero without contact")
        result["frame_neoforce_features"] = frame_neo[slice(start, end)].float()
    return result
