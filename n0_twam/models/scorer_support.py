"""Query-local scored history selection for frozen FM reward evaluation.

The existing retention plan owns all earlier phases. Only the final noisy
video/action phases use the scorer's full observed-history selection.
"""
import torch


def apply_scorer_support(plan, memory, spec):
    layout = memory["layout"]
    seq, phase, clean, kind = (layout[k] for k in ("seq", "phase", "clean", "kind"))
    if torch.unique(seq[seq >= 0]).tolist() != [0]:
        raise ValueError("Scorer rewards require one episode per forward")
    target = int(spec["target_phase"])
    if target < 2 or target % 2 or target != int(phase[seq >= 0].max()) - 1:
        raise ValueError("Only the final video/action frame pair is supported")
    historical = (seq == 0) & clean & (phase < target)
    video = ((seq == 0) & clean & (kind == 0)).nonzero().flatten()
    scores = torch.as_tensor(spec["video_scores"], device=seq.device, dtype=torch.float32)
    if scores.shape != video.shape or not torch.isfinite(scores).all():
        raise ValueError("Scores must cover the exact clean-video token layout")
    budgets = tuple(spec["budgets"])
    if len(budgets) != 3 or any(type(k) is not int or k < 0 for k in budgets):
        raise ValueError("Three nonnegative modality history budgets required")
    if sum(budgets) != int(memory["config"]["capacity"]):
        raise ValueError("History budgets must sum to the fixed group capacity")
    rank_scores = memory["rows"]["world_time_id"].float().clone()
    rank_scores[video] = scores
    selected = torch.zeros_like(clean)
    for modality, budget in enumerate(budgets):
        eligible = (historical & (kind == modality)).nonzero().flatten()
        # Stable sequence-address tie breaking. Action/tactile remain recency
        # controls until the scorer's schema includes these modalities.
        order = torch.argsort(rank_scores[eligible], descending=True, stable=True)
        selected[eligible[order[:budget]]] = True
    result = dict(plan, noisy=plan["noisy"].clone())
    result["noisy"][target:] = selected
    memory["config"]["scorer_selected_indices"] = selected.nonzero().flatten().detach().cpu().tolist()
    return result
