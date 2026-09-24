"""Causal motion support for N0-TWAM's two teacher-forced video branches."""
import torch

FIELDS = ("rgb_motion_patch_indices", "rgb_motion_valid_mask",
          "world_time_id", "dino_features", "neoforce_features",
          "observation_flag", "visual_valid", "tactile_valid", "motion_scores")

def prepare_causal_motion(latent, chunk_size):
    """Keep observed support for clean conditions; carry past support to targets.

    The first prediction chunk has no prior observation and is fully masked.
    No future target mask, feature or target-dependent patch count is revealed.
    K is fixed by the original loader. Content still comes from the appropriate
    target/condition frame and uses that frame's original RoPE coordinates.
    """
    if "rgb_motion_patch_indices" not in latent:
        raise ValueError("causal motion training requires loader [B,F,K] patch indices")
    original = {k: latent[k] for k in FIELDS if k in latent}
    indices = original["rgb_motion_patch_indices"]
    if indices.ndim != 3 or chunk_size < 1:
        raise ValueError("motion indices must be [B,F,K]; chunk_size must be positive")
    b,frames,k = indices.shape
    source = (torch.arange(frames,device=indices.device)//chunk_size)*chunk_size-1
    cold = source < 0
    source = source.clamp_min(0)
    latent["condition_motion"] = original
    for key,value in original.items():
        if value.shape[:3] != indices.shape:
            raise ValueError(f"{key} must align with [B,F,K] motion addresses")
        target = value[:,source].clone()
        target[:,cold] = -1 if key in ("rgb_motion_patch_indices","world_time_id") else 0
        if key == "world_time_id":
            # Preserve actual target grid time, not the age of the support.
            times = latent["grid_id"][:,0].reshape(b,frames,-1)[:,:,0]
            target = times[:,:,None].expand(b,frames,k).to(value.dtype).clone()
            valid = original.get("rgb_motion_valid_mask",indices>=0)[:,source].clone()
            valid[:,cold]=False
            target.masked_fill_(~valid,-1)
        elif key == "observation_flag":
            target.zero_()
        elif key in ("neoforce_features","tactile_valid"):
            # No predicted NeoForce is fabricated from a real contact.
            target.zero_()
        latent[key]=target

    # A tactile-only address has no visual prediction index. Keep it in the
    # observed condition branch, but do not invent DINO/NeoForce for prediction.
    if "visual_valid" in original:
        valid = latent["rgb_motion_valid_mask"] & latent["visual_valid"].bool()
        latent["rgb_motion_valid_mask"] = valid
        for key in FIELDS:
            if key not in latent:
                continue
            value = latent[key]
            mask = valid if value.ndim == 3 else valid[..., None]
            latent[key] = value.masked_fill(~mask, -1 if key in (
                "rgb_motion_patch_indices", "world_time_id") else 0)
