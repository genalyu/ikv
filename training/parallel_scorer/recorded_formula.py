"""User-recorded visual formula. All times must use the same world-step unit."""
import torch

def recorded_visual_formula(times, reference_time, contact_duration, observed,
                            similarity, class_last_time, feature_present,
                            scale=8.0, threshold=0.9):
    """No direct cosine reward, persistence reward, task term or random mixture.

    Callers must supply class matches from their declared content dictionary.
    No raw-frame/seconds conversion is performed implicitly.
    """
    arrays=(times,contact_duration,observed,similarity,class_last_time,feature_present)
    if any(x.shape != times.shape for x in arrays):
        raise ValueError("Formula inputs must be aligned")
    if scale <= 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid scale or threshold")
    if any(not torch.isfinite(x).all() for x in (times,contact_duration,similarity,class_last_time)):
        raise ValueError("Nonfinite formula input")
    if (contact_duration < 0).any():
        raise ValueError("Negative contact duration")
    recency=torch.exp(-(reference_time-times).clamp_min(0)/scale)
    contact=torch.exp(-contact_duration/scale)
    valid=observed.bool() & feature_present.bool() & (similarity >= threshold)
    content=torch.exp(-(class_last_time-times).clamp_min(0)/scale)*valid
    return dict(score=recency+contact+content,time=recency,contact=contact,class_recency=content)

