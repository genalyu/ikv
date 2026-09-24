"""N0-TWAM teacher-forced training with the serving IKV retention policy.

Only backbone attention execution changes. Inputs, condition corruption,
denoising targets and losses remain owned by the original Trainer/model.
No persistent inference pool or detached historical K/V is used here.
"""
import torch
from .global_kv_retention import GlobalKVRetention, RetentionConfig


def training_metadata(grid, layout, splits, latent, action, motion_layout):
    """Describe condition tokens; index features never enter content embeddings."""
    n = len(layout["seq"])
    device = grid.device
    full_grid = torch.zeros(4, n, device=device)
    full_grid[:, :grid.shape[-1]] = grid[0]
    rows = dict(world_time_id=full_grid[0].float(),
                grid_position=full_grid[1:].T.float(), kind=layout["kind"],
                observation_flag=torch.ones(n, dtype=torch.bool, device=device),
                duration=torch.ones(n, device=device))
    v_start, v_end = splits[0], sum(splits[:2])
    for name, source in (("dino", "dino_features"), ("neoforce", "neoforce_features")):
        # Sparse fields have already been validated/aligned by _rgb_motion_layout.
        features = None
        if motion_layout is not None and motion_layout["semantic_index"] is not None:
            features = motion_layout["semantic_index"][name]
        elif motion_layout is None and source in latent:
            features = latent[source]
        if features is None:
            rows[name] = torch.empty(n, 0, device=device)
        else:
            features = features.reshape(v_end-v_start, features.shape[-1]).detach().float()
            if not torch.isfinite(features).all():
                raise ValueError("IKV training metadata must be finite")
            rows[name] = torch.zeros(n, features.shape[-1], device=device)
            rows[name][v_start:v_end] = features
    values = action["latent"].permute(0, 2, 3, 4, 1).flatten(0, 3).detach().float()
    start = sum(splits[:3])
    rows["action"] = torch.zeros(n, values.shape[-1], device=device)
    if len(values) != splits[3]:
        raise ValueError("condition action metadata/token count mismatch")
    rows["action"][start:start+splits[3]] = values
    return rows


def run_ikv_training(mot, hidden, text, timestep, temb, rope, memory):
    """Execute the original clean/noisy causal phases with bounded condition KV.

    The policy owns discrete, detached decisions. Every retained layer K/V keeps
    its autograd connection. Immutable per-phase contexts make activation
    checkpoint recomputation independent of later retention updates.
    """
    config, layout, rows = memory["config"], memory["layout"], memory["rows"]
    capacity = config["capacity"]
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("ikv_train_capacity must be a positive integer")
    retention = RetentionConfig(**config.get("retention", {}))
    seq, phase, clean, kind = (layout[k] for k in ("seq", "phase", "clean", "kind"))
    valid_seqs = torch.unique(seq[seq >= 0]).tolist()
    if not valid_seqs:
        raise ValueError("IKV training needs valid tokens")
    batches = len(valid_seqs)
    if valid_seqs != list(range(batches)):
        raise ValueError("packed batch IDs must be contiguous")
    if text is not None and text.shape[1] % batches:
        raise ValueError("text tokens must divide evenly across packed samples")
    outputs, addresses = [], []
    for batch in valid_seqs:
        policy = GlobalKVRetention(capacity, hidden.device, retention)
        mask = torch.zeros(capacity, dtype=torch.bool, device=hidden.device)
        slots = torch.empty(0, dtype=torch.long, device=hidden.device)
        past = [None] * mot.num_layers
        own_text = None if text is None else text.chunk(batches, dim=1)[batch]
        for stage in torch.unique(phase[seq == batch], sorted=True).tolist():
            positions = ((seq == batch) & (phase == stage)).nonzero().flatten()
            is_clean = clean[positions]
            clean_positions = is_clean.nonzero().flatten()
            source = positions[clean_positions]
            incoming = {k: v[source] for k, v in rows.items()}
            with torch.no_grad():
                # Do not let same-phase clean DINO/labels determine which past
                # the noisy prediction can see. The anchor is committed history.
                provisional = dict(incoming, observation_flag=torch.zeros_like(incoming["observation_flag"]))
                new_slots, victims = policy.plan(mask, len(source), incoming)
                _, noisy_victims = policy.plan(mask, int((~is_clean).sum()), provisional)
                clean_keep = ~torch.isin(slots, victims)
                noisy_keep = ~torch.isin(slots, noisy_victims)
                keep = clean_keep | noisy_keep
                kept_slots = slots[clean_keep]
            old = [None if kv is None else (kv[0][:, keep], kv[1][:, keep]) for kv in past]
            clean_old = clean_keep[keep]
            history_mask = torch.where(is_clean[:,None], clean_keep[None,keep], noisy_keep[None,keep])
            # Original N0-TWAM mask: clean↔clean within phase, noisy↔noisy
            # within phase, both can read earlier condition history.
            current_mask = is_clean[:, None] == is_clean[None, :]
            attention_mask = torch.cat((
                history_mask,
                current_mask), dim=1)[None, None]
            context = dict(past=old, mask=attention_mask, clean_positions=clean_positions)
            lengths = [int((kind[positions] == k).sum()) for k in range(3)]
            v, a, t = lengths
            slices = [("video", 0, v), ("action", v, v+a), ("tactile", v+a, v+a+t)]
            # No packed cross-sample mask is needed after selecting this sample.
            mot.set_masks(self_block_mask=None, cross_masks={})
            result, current = mot(hidden[:, positions], own_text, timestep[:, positions],
                                  temb[:, positions], rope[:, positions], slices,
                                  training_context=context)
            outputs.append(result)
            addresses.append(positions)
            combined_slots = torch.cat((kept_slots, new_slots))
            past = [(k if previous is None else torch.cat((previous[0][:, clean_old], k), dim=1),
                     v if previous is None else torch.cat((previous[1][:, clean_old], v), dim=1))
                    for previous, (k, v, _q) in zip(old, current)]
            with torch.no_grad():
                old_mask = mask.clone()
                mask[victims] = False
                mask[new_slots] = True
                policy.commit(new_slots, incoming, old_mask)
                # Same policy usage estimator and all-layer averaging as serving;
                # only condition queries count, never artificially noisy targets.
                measurements = [policy.measure_usage(q, kv[0].detach(), combined_slots)
                                for (_k, _v, q), kv in zip(current, past) if q.shape[1]]
                policy.add_usage(measurements)
            slots = combined_slots
    result = torch.zeros_like(hidden)
    return result.index_copy(1, torch.cat(addresses), torch.cat(outputs, dim=1))
