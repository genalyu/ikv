"""Validate native RGB and normalize to float NCHW without channel swaps."""
import torch
def rgb_tensor(value):
    x=torch.as_tensor(value)
    if x.ndim==3:x=x.unsqueeze(0)
    if x.ndim!=4 or (x.shape[1]==3)==(x.shape[-1]==3):
        raise ValueError("RGB must unambiguously be [T,3,H,W] or [T,H,W,3]")
    if x.shape[-1]==3:x=x.permute(0,3,1,2)
    if x.dtype==torch.uint8:x=x.float()/255
    elif not x.is_floating_point():raise ValueError("RGB must be uint8 or float in [0,1]")
    if not torch.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError("RGB must be finite in [0,1]")
    return x.float()
