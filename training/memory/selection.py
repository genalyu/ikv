"""Causal streaming eviction shared by offline teacher forcing and serving.

Addresses are native KV slots, never scorer-region indices. The caller supplies
scores computed from observations available BEFORE the incoming phase. No task
labels, targets or future RGB are accepted here.
"""
import torch

VERSION = "memory-causal-evict-v3"


def scorer_to_native(values):
    """[F,camera,row,col,...] -> [F,row,camera,col,...]."""
    if values.ndim < 4 or values.shape[1:4] != (3, 8, 8):
        raise ValueError("Expected three cameras with native 8x8 patches")
    order = (0, 2, 1, 3, *range(4, values.ndim))
    return values.permute(order).reshape(values.shape[0], 192, *values.shape[4:])


def evict(live_ids, live_kinds, scores, incoming_counts, budgets):
    """Reserve current tokens, retain only LIVE history, stable-ID ties.

    The returned IDs are the only old KV that may survive this phase. Evicted
    IDs cannot re-enter unless the caller explicitly re-encodes an observation.
    """
    ids = torch.as_tensor(live_ids, dtype=torch.long)
    kinds = torch.as_tensor(live_kinds, device=ids.device, dtype=torch.long)
    scores = torch.as_tensor(scores, device=ids.device, dtype=torch.float32)
    if ids.ndim != 1 or kinds.shape != ids.shape or scores.shape != ids.shape:
        raise ValueError("One score and modality per live native KV slot required")
    if len(ids.unique()) != len(ids) or (ids < 0).any() or not torch.isfinite(scores).all():
        raise ValueError("Invalid native IDs or nonfinite live scores")
    if ((kinds < 0) | (kinds > 1)).any():
        raise ValueError("Memory RGB-only supports video=0/action=1")
    if len(budgets) != 2 or len(incoming_counts) != 2:
        raise ValueError("Explicit video/action budgets required")
    kept = []
    for kind, (budget, incoming) in enumerate(zip(budgets, incoming_counts)):
        if type(budget) is not int or type(incoming) is not int or incoming < 0 or budget < incoming:
            raise ValueError("Current phase exceeds modality capacity")
        ix = (kinds == kind).nonzero().flatten()
        ix = ix[torch.argsort(ids[ix], stable=True)]
        ix = ix[torch.argsort(scores[ix], descending=True, stable=True)]
        kept.append(ids[ix[:budget-incoming]])
    return torch.cat(kept).sort().values


def replay(frames, score_prefix, budgets=(3584, 512), video_per_frame=192, action_per_frame=12, chunk_size=2):
    """Build [2*ceil(F/chunk_size),K] native clean-token IDs (-1 padded) without future access.

    score_prefix(n) must score exactly frames [0,n), in native spatial order.
    Both phases for target f may consult only frames before f. The action
    chunk drives toward target video f, so its real RGB is NOT yet observed.
    A newly generated video phase fits its reserved slots; action insertion
    never triggers video eviction, so no target-frame score is needed.
    """
    if frames < 1 or video_per_frame != 192 or action_per_frame < 1 or chunk_size not in (1,2):
        raise ValueError("Invalid RGB memory geometry")
    V = frames * video_per_frame
    live = torch.empty(0, dtype=torch.long)
    stages = 2*((frames+chunk_size-1)//chunk_size)
    plan = torch.full((stages, sum(budgets)), -1, dtype=torch.long)
    cached = {}
    for stage in range(stages):
        group, kind = divmod(stage, 2)
        frame = group*chunk_size
        width = min(chunk_size, frames-frame)
        prefix = frame
        scores = torch.zeros(len(live))
        video = live < V
        incoming = (width*video_per_frame, 0) if kind == 0 else (0, width*action_per_frame)
        if int(video.sum()) + incoming[0] > budgets[0]:
            if (live[video] >= prefix*video_per_frame).any():
                raise ValueError("Cannot score unobserved target RGB")
            if prefix not in cached:
                value = torch.as_tensor(score_prefix(prefix)).float().cpu().flatten()
                if len(value) != prefix*video_per_frame or not torch.isfinite(value).all():
                    raise ValueError("Scorer must return finite scores for exact observed prefix")
                cached = {prefix: value}
            scores[video] = cached[prefix][live[video]]
        scores[~video] = live[~video].float()  # action history remains recency
        live = evict(live, (live >= V).long(), scores, incoming, tuple(budgets))
        plan[stage, :len(live)] = live
        start = frame*video_per_frame if kind == 0 else V+frame*action_per_frame
        count = incoming[kind]
        live = torch.cat((live, torch.arange(start, start+count)))
    return plan


def support_from_plan(layout, selected, capacity):
    """Map canonical clean IDs to the actual transformer token layout."""
    seq, phase, clean, kind = (layout[k] for k in ("seq", "phase", "clean", "kind"))
    if torch.unique(seq[seq >= 0]).tolist() != [0]:
        raise ValueError("Memory training requires one complete episode per rank")
    selected = torch.as_tensor(selected, device=seq.device, dtype=torch.long)
    if selected.ndim == 3 and selected.shape[0] == 1:
        selected = selected[0]
    stages = int(phase[seq >= 0].max())+1
    if selected.shape != (stages, capacity):
        raise ValueError("Plan/phase/capacity mismatch; crops and random chunk widths are forbidden")
    addresses = torch.cat([((seq == 0) & clean & (kind == k)).nonzero().flatten() for k in (0, 1)])
    if ((seq == 0) & (kind == 2)).any():
        raise ValueError("Tactile tokens are forbidden in RGB Memory plan")
    valid = selected >= 0
    if (selected < -1).any() or (selected[valid] >= len(addresses)).any():
        raise ValueError("Memory plan contains invalid native KV address")
    mapped = addresses[selected.clamp_min(0)]
    rows = torch.arange(stages, device=seq.device)[:, None].expand_as(selected)
    if (phase[mapped[valid]] >= rows[valid]).any():
        raise ValueError("Future/same-phase clean KV leakage in Memory plan")
    support = torch.zeros(stages, len(seq), dtype=torch.bool, device=seq.device)
    support[rows[valid], mapped[valid]] = True
    return dict(clean=support, noisy=support)
