"""Inference-like current-observation local tactile for final-frame FM rewards."""
import torch


def observed_local_mask(action_shape, tactile_shape, observed_frame, device):
    _, _, frames, actions, width = action_shape
    _, sensors, tactile_frames, height, tactile_width = tactile_shape
    if tactile_frames != frames or frames < 2 or observed_frame != frames - 2:
        raise ValueError("Local tactile must use exactly the last observed frame")
    query = torch.arange(frames, device=device).repeat_interleave(actions * width)
    keys = torch.arange(frames, device=device).repeat_interleave(height * tactile_width).repeat(sensors)
    # Earlier outputs keep their own current observation; final reward uses
    # the last observation, never future expert tactile or all local history.
    return (keys[None, :] == query.clamp(max=observed_frame)[:, None])[None, None]
