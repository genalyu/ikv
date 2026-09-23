"""Experimental RGB-only feature correspondence, NOT geometric ground truth.

NeoForce is not explicitly trained for point correspondence. Mutual cosine
matches are a configurable hypothesis to evaluate, not proof of contact location
or visibility. Low confidence always falls back to tactile-only. No depth or
simulator object poses are consumed.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def mutual_contact_matches(visual, tactile, response, *, min_similarity=.7, min_margin=.15):
    """[V,D], [M,D], [M] -> [M] visual row IDs (-1 means unmatched).

    The per-sensor caller decides grid layout and camera identity separately.
    Scores are cosine similarities, never called contact probabilities.
    """
    if (visual.ndim != 2 or tactile.ndim != 2 or visual.shape[1] != tactile.shape[1]
            or visual.shape[1] == 0 or response.shape != (len(tactile),)):
        raise ValueError('visual/tactile/response shapes must be [V,D]/[M,D]/[M]')
    if any(not torch.isfinite(x).all() for x in (visual, tactile, response)) or (response < 0).any():
        raise ValueError('matching inputs must be finite, response nonnegative')
    if (not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1
            or not math.isfinite(min_margin) or not 0 < min_margin <= 2):
        raise ValueError('matching thresholds out of range')
    rows = torch.full((len(tactile),), -1, device=tactile.device, dtype=torch.long)
    scores = tactile.new_zeros(len(tactile))
    margins = tactile.new_zeros(len(tactile))
    active = ((response > 0) & tactile.norm(dim=-1).gt(0)).nonzero().flatten()
    if len(visual) < 2 or not len(active):
        return {'visual_rows': rows, 'scores': scores, 'margins': margins}
    cosine = F.normalize(tactile[active].float(), dim=-1) @ F.normalize(visual.float(), dim=-1).T
    cosine[:, visual.norm(dim=-1) == 0] = -torch.inf
    top, indices = cosine.topk(2, dim=1)
    margin = top[:, 0] - top[:, 1]
    reciprocal = cosine.argmax(dim=0)[indices[:, 0]] == torch.arange(len(active), device=active.device)
    # Reciprocal ties in the tactile direction must also be rejected.
    if len(active) > 1:
        reverse = cosine.topk(2, dim=0).values
        reciprocal &= (reverse[0] - reverse[1])[indices[:, 0]] >= min_margin
    valid = (reciprocal & torch.isfinite(top).all(dim=1)
             & (top[:, 0] >= min_similarity) & (margin >= min_margin))
    rows[active[valid]] = indices[valid, 0]
    scores[active] = top[:, 0].nan_to_num(neginf=-1.)
    margins[active] = margin.nan_to_num(nan=0., posinf=0., neginf=0.)
    return {'visual_rows': rows, 'scores': scores, 'margins': margins}


def pool_visual_tokens(tokens, target_size):
    """[B,T,H,W,D] -> [B,T,target_h,target_w,D], camera-local only."""
    b, t, h, w, d = tokens.shape
    pooled = F.adaptive_avg_pool2d(tokens.permute(0, 1, 4, 2, 3).reshape(b*t, d, h, w), target_size)
    return pooled.reshape(b, t, d, *target_size).permute(0, 1, 3, 4, 2)
