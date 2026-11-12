"""Stage-1 sparse-history heuristic in world-time steps."""
import torch


def sample_training_weights(*, generator=None, device="cpu"):
    """Independent Uniform[0, 2] draws (mean one), using the training RNG."""
    return 2.0 * torch.rand(3, generator=generator, device=device)


def heuristic_scores(times, decision_time, is_visual, contact_duration,
                     class_recency, *, weights=(1., 1., 1.)):
    """Raw weighted sum; class_recency is the gated prototype term.

    No normalization, score noise, seconds conversion or prompt relevance.
    Missing contact must be represented by zero.
    """
    times = torch.as_tensor(times).float()
    contact_duration = torch.as_tensor(contact_duration, device=times.device).float()
    class_recency = torch.as_tensor(class_recency, device=times.device).float()
    is_visual = torch.as_tensor(is_visual, device=times.device, dtype=torch.bool)
    weights = torch.as_tensor(weights, device=times.device, dtype=torch.float32)
    if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("heuristic weights must be three finite nonnegative values")
    if any(x.shape != times.shape for x in (contact_duration, class_recency, is_visual)):
        raise ValueError("heuristic metadata must have aligned shapes")
    anchor = torch.as_tensor(decision_time, device=times.device).float()
    if (anchor.numel() != 1 or not torch.isfinite(anchor).all()
            or not torch.isfinite(times).all() or not torch.isfinite(contact_duration).all()
            or not torch.isfinite(class_recency).all() or (contact_duration < 0).any()
            or ((class_recency < 0) | (class_recency > 1)).any()):
        raise ValueError("invalid heuristic time, duration or class-recency metadata")
    recency = torch.exp(-(anchor-times).clamp_min(0) / 8.)
    contact = is_visual * torch.exp(-contact_duration / 8.)
    return weights[0] * recency + weights[1] * contact + weights[2] * class_recency
