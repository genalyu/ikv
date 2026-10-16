"""N0-TWAM teacher-forced training with the serving IKV retention policy.

Only backbone attention execution changes. Inputs, condition corruption,
denoising targets and losses remain owned by the original Trainer/model.
No persistent inference pool or detached historical K/V is used here.
"""
from dataclasses import replace
import os

import torch
import torch.distributed as dist
from .multimodal_kv_retention import make_retention_policy
from .global_kv_retention import GlobalKVRetention, RetentionConfig


_HISTORY_DECISION_CACHE = {}


def training_metadata(grid, layout, splits, latent, action, motion_layout, version=1):
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
        if version == 2 and name == "dino" and "dense_dino_features" in latent:
            features = latent["dense_dino_features"].flatten(1, 2)
            if motion_layout is not None:
                indices = motion_layout["indices"]
                features = features.gather(1, indices[..., None].expand(-1, -1, features.shape[-1]))
        if features is None:
            rows[name] = torch.empty(n, 0, device=device)
        else:
            features = features.reshape(v_end-v_start, features.shape[-1]).detach().float()
            if not torch.isfinite(features).all():
                raise ValueError("IKV training metadata must be finite")
            rows[name] = torch.zeros(n, features.shape[-1], device=device)
            rows[name][v_start:v_end] = features
    if version == 2 and "task_relevance" in latent:
        relevance = latent["task_relevance"].flatten(1, 2)
        if motion_layout is not None:
            relevance = relevance.gather(1, motion_layout["indices"])
        relevance = relevance.reshape(-1).detach().float()
        if len(relevance) != v_end-v_start or not torch.isfinite(relevance).all() or ((relevance < 0) | (relevance > 1)).any():
            raise ValueError("training task relevance must align with video patches in [0,1]")
        rows["task_relevance"] = torch.zeros(n, device=device)
        rows["task_relevance"][v_start:v_end] = relevance
    values = action["latent"].permute(0, 2, 3, 4, 1).flatten(0, 3).detach().float()
    start = sum(splits[:3])
    rows["action"] = torch.zeros(n, values.shape[-1], device=device)
    if len(values) != splits[3]:
        raise ValueError("condition action metadata/token count mismatch")
    rows["action"][start:start+splits[3]] = values
    if version == 2:
        frame_neo = latent.get("frame_neoforce_features")
        if frame_neo is not None:
            if frame_neo.ndim != 3 or not torch.isfinite(frame_neo).all():
                raise ValueError("frame_neoforce_features must be finite [B,F,D], zero without contact")
            rows["neoforce"] = torch.zeros(n, frame_neo.shape[-1], device=device)
            selected = layout["clean"] & (layout["seq"] >= 0) & (layout["kind"] != 1)
            frame = rows["world_time_id"][selected].long()
            if len(frame) and (frame.min() < 0 or frame.max() >= frame_neo.shape[1]):
                raise ValueError("NeoForce frames do not cover the training sequence")
            rows["neoforce"][selected] = frame_neo[layout["seq"][selected], frame]
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
    if retention.task_weight and "task_relevance" not in rows:
        raise ValueError("task_weight requires regenerated semantic feature sidecars")
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
    local_phases = sum(len(torch.unique(phase[seq == batch])) for batch in valid_seqs)
    max_phases = local_phases
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if batches != 1:
            raise ValueError("Distributed IKV currently requires per-rank batch_size=1")
        count = torch.tensor(local_phases, dtype=torch.long, device=hidden.device)
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        max_phases = int(count.item())
    for batch in valid_seqs:
        policy = make_retention_policy(capacity, hidden.device, retention)
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
                if retention.version == 2:
                    noisy_source = positions[~is_clean]
                    # Allocation needs only kinds/times. No clean semantic metadata
                    # may influence simultaneous noisy-branch history selection.
                    provisional = {k: v[noisy_source].clone() for k, v in rows.items()}
                    provisional["observation_flag"].zero_()
                    provisional["dino"].zero_()
                    provisional["neoforce"].zero_()
                    if "contact_present" in provisional:
                        provisional["contact_present"].zero_()
                else:
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
                if retention.version == 2:
                    dense = memory.get("dense_dino_features")
                    if (dense is None or not dense.shape[-1]) and (
                            retention.persistence_weight or retention.class_recency_weight):
                        raise ValueError("v2 content scores require full-grid dense_dino_features")
                    if dense is not None:
                        # Update only after this phase has evaluated. No future
                        # observation can affect its simultaneous noisy branch.
                        real_times = incoming["world_time_id"][incoming["kind"] != 1]
                        if len(real_times):
                            last = int(real_times.max())
                            frame_ids = torch.arange(min(last + 1, dense.shape[1]), device=hidden.device)
                            features = dense[batch, frame_ids]
                            times = frame_ids[:, None].expand(features.shape[:2]).flatten().float()
                            policy.observe_dense(features.flatten(0, 1), times, torch.ones_like(times))
            slots = combined_slots
    result = torch.zeros_like(hidden)
    result = result.index_copy(1, torch.cat(addresses), torch.cat(outputs, dim=1))
    # FSDP2 collectives require identical MoT calls on every rank. Variable
    # episode lengths yield different phase counts even with a shared chunk
    # width. Zero-weight dummy phases preserve the data loss and gradients.
    if max_phases > local_phases:
        first = (seq == valid_seqs[0]).nonzero().flatten()[:1]
        repeated = first.repeat(3)
        dummy_slices = [("video", 0, 1), ("action", 1, 2), ("tactile", 2, 3)]
        dummy_context = dict(past=[None] * mot.num_layers,
                             mask=torch.ones(1, 1, 3, 3, dtype=torch.bool, device=hidden.device),
                             clean_positions=torch.arange(3, device=hidden.device))
        dummy_text = None if text is None else text.chunk(batches, dim=1)[0]
        zero_anchor = hidden.new_zeros(())
        for _ in range(max_phases - local_phases):
            mot.set_masks(self_block_mask=None, cross_masks={})
            dummy, _ = mot(hidden[:, repeated], dummy_text, timestep[:, repeated],
                           temb[:, repeated], rope[:, repeated], dummy_slices,
                           training_context=dummy_context)
            zero_anchor = zero_anchor + dummy.sum() * 0.0
        result = result + zero_anchor
    return result


def sample_ikv_capacity(memory, device, retention):
    """Draw K without consulting future token counts or feature values."""
    config = memory["config"]
    maximum = config["capacity"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("IKV capacity must be a positive integer")
    if not config.get("sample_capacity", False):
        config["sampled_capacity"] = maximum
        return maximum, retention
    minimum = config.get("min_capacity", (maximum + 1) // 2)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1 or minimum > maximum:
        raise ValueError("IKV min_capacity must be in [1, capacity]")
    if retention.version == 2:
        limits = [retention.video_capacity, retention.action_capacity, retention.tactile_capacity]
        if sum(limits) != maximum:
            raise ValueError("v2 modality capacities must sum to IKV capacity")
        if minimum < 3:
            raise ValueError("v2 min_capacity must reserve one slot per modality")
    sampled = int(torch.randint(minimum, maximum + 1, (1,), device=device))
    config["sampled_capacity"] = sampled
    if retention.version == 1:
        return sampled, retention
    # Reserve one slot in each modality, then allocate the remaining sampled
    # slots in proportion to configured modality slack. No trajectory metadata
    # enters the budget decision; an oversized current phase fails explicitly.
    slack = [limit - 1 for limit in limits]
    total_slack = sum(slack)
    extra = sampled - 3
    shares = [divmod(extra * room, total_slack) if total_slack else (0, 0)
              for room in slack]
    budgets = [1 + share[0] for share in shares]
    leftover = sampled - sum(budgets)
    order = sorted(range(3), key=lambda i: (-shares[i][1], i))
    for i in order:
        if not leftover:
            break
        if budgets[i] < limits[i]:
            budgets[i] += 1
            leftover -= 1
    assert leftover == 0 and sum(budgets) == sampled
    config["sampled_modality_capacities"] = budgets
    return sampled, replace(retention, video_capacity=budgets[0],
                            action_capacity=budgets[1], tactile_capacity=budgets[2])


def build_ikv_support_plan(memory, device, *, return_groups=False):
    """Replay the detached retention policy from recorded metadata only.

    Attention-query usage must be disabled for this path; otherwise the mask
    would depend on a model pass and could not be prepared before training.
    """
    config, layout, rows = memory["config"], memory["layout"], memory["rows"]
    retention = RetentionConfig(**config.get("retention", {}))
    if retention.task_weight and "task_relevance" not in rows:
        raise ValueError("task_weight requires regenerated semantic feature sidecars")
    if any(getattr(retention, name) for name in
           ("query_weight", "action_query_weight", "tactile_query_weight")):
        raise ValueError("Masked IKV requires zero query-usage score weights")
    seq, phase, clean = (layout[name] for name in ("seq", "phase", "clean"))
    valid_seqs = torch.unique(seq[seq >= 0]).tolist()
    if valid_seqs != [0]:
        raise ValueError("Masked IKV currently requires per-rank batch_size=1")
    capacity, retention = sample_ikv_capacity(memory, device, retention)
    groups = []
    if not return_groups:
        support_clean = torch.zeros(int(phase.max()) + 1, len(seq), dtype=torch.bool, device=device)
        support_noisy = torch.zeros_like(support_clean)
    policy = make_retention_policy(capacity, device, retention)
    history_key = config.get("history_cache_key")
    if history_key is not None and retention.version == 2:
        cache_key = (int(history_key), retention.content_capacity,
                     retention.content_threshold)
        if cache_key not in _HISTORY_DECISION_CACHE:
            if len(_HISTORY_DECISION_CACHE) >= 512:
                _HISTORY_DECISION_CACHE.pop(next(iter(_HISTORY_DECISION_CACHE)))
            _HISTORY_DECISION_CACHE[cache_key] = {}
        policy.history.decision_cache = _HISTORY_DECISION_CACHE[cache_key]
    occupied = torch.zeros(capacity, dtype=torch.bool, device=device)
    slots = torch.empty(0, dtype=torch.long, device=device)
    slot_to_token = torch.full((capacity,), -1, dtype=torch.long, device=device)
    for stage in torch.unique(phase[seq == 0], sorted=True).tolist():
        positions = ((seq == 0) & (phase == stage)).nonzero().flatten()
        is_clean = clean[positions]
        source = positions[is_clean]
        noisy_source = positions[~is_clean]
        incoming = {name: value[source] for name, value in rows.items()}
        if retention.version == 2:
            provisional = {name: value[noisy_source].clone() for name, value in rows.items()}
            provisional["observation_flag"].zero_()
            provisional["dino"].zero_()
            provisional["neoforce"].zero_()
            if "contact_present" in provisional:
                provisional["contact_present"].zero_()
        else:
            provisional = dict(incoming,
                               observation_flag=torch.zeros_like(incoming["observation_flag"]))
        new_slots, victims = policy.plan(occupied, len(source), incoming)
        _, noisy_victims = policy.plan(occupied, len(noisy_source), provisional)
        clean_keep = ~torch.isin(slots, victims)
        noisy_keep = ~torch.isin(slots, noisy_victims)
        old_tokens = slot_to_token[slots]
        if return_groups:
            # Every occupied token is clean and belongs to an earlier phase.
            # Preserve the dense-plan key order exactly, without a [phase, token] table.
            for query, keep in ((source, clean_keep), (noisy_source, noisy_keep)):
                if query.numel():
                    keys = torch.cat((old_tokens[keep], query)).sort().values
                    groups.append((query, keys))
        else:
            support_clean[int(stage), old_tokens[clean_keep]] = True
            support_noisy[int(stage), old_tokens[noisy_keep]] = True
        kept_slots = slots[clean_keep]
        old_mask = occupied.clone()
        occupied[victims] = False
        occupied[new_slots] = True
        policy.commit(new_slots, incoming, old_mask)
        slot_to_token[new_slots] = source
        if retention.version == 2:
            dense = memory.get("dense_dino_features")
            if (dense is None or not dense.shape[-1]) and (
                    retention.persistence_weight or retention.class_recency_weight):
                raise ValueError("v2 content scores require full-grid dense DINO")
            if dense is not None:
                real_times = incoming["world_time_id"][incoming["kind"] != 1]
                if len(real_times):
                    last = int(real_times.max())
                    # History already committed earlier dense frames. Replaying the
                    # growing prefix only scans duplicates and synchronizes on every
                    # old timestamp; retain the exact chronological new observations.
                    first = 0 if policy.history.clock is None else int(policy.history.clock) + 1
                    stop = min(last + 1, dense.shape[1])
                    if first < stop:
                        frame_ids = torch.arange(first, stop, device=device)
                        features = dense[0, frame_ids]
                        times = frame_ids[:, None].expand(features.shape[:2]).flatten().float()
                        policy.observe_dense(features.flatten(0, 1), times, torch.ones_like(times))
        slots = torch.cat((kept_slots, new_slots))
    if history_key is not None and retention.version == 2 and os.getenv("IKV_HISTORY_CACHE_DEBUG") == "1":
        print(f"IKV_HISTORY_CACHE key={history_key} hits={policy.history.cache_hits} "
              f"frames={len(policy.history.decision_cache)}", flush=True)
    return dict(groups=groups) if return_groups else dict(clean=support_clean, noisy=support_noisy)


def _preserve_parent_full_blocks(mask):
    """Keep 128-block traversal order while pruning empty 64 blocks.

    Newly full child blocks must stay in the partial loop: moving them to the
    full loop changes softmax reduction order and can amplify BF16 rounding.
    """
    from torch.nn.attention.flex_attention import BlockMask

    if mask.BLOCK_SIZE != (64, 64) or mask.full_kv_num_blocks is None:
        return mask

    def dense(count, indices):
        columns = torch.arange(indices.shape[-1], device=indices.device)
        flags = (columns < count[..., None]).to(torch.int32)
        return torch.zeros_like(indices, dtype=torch.int32).scatter_add(
            -1, indices.long(), flags
        ).bool()

    partial = dense(mask.kv_num_blocks, mask.kv_indices)
    full = dense(mask.full_kv_num_blocks, mask.full_kv_indices)
    nq, nk = full.shape[-2:]
    padded = torch.nn.functional.pad(full, (0, nk % 2, 0, nq % 2))
    parent = padded.reshape(
        *full.shape[:-2], (nq + 1) // 2, 2, (nk + 1) // 2, 2
    ).all(-1).all(-2)
    kept_full = parent.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :nq, :nk]
    refined_partial = (partial | full) & ~kept_full

    def ordered(blocks):
        blocks = blocks.to(torch.int32)
        counts = blocks.sum(-1).to(torch.int32)
        indices = blocks.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
        return counts, indices

    partial_counts, partial_indices = ordered(refined_partial)
    full_counts, full_indices = ordered(kept_full)
    return BlockMask.from_kv_blocks(
        partial_counts, partial_indices, full_counts, full_indices,
        BLOCK_SIZE=64, mask_mod=mask.mask_mod, seq_lengths=mask.seq_lengths,
    )


def run_ikv_masked_training(mot, hidden, text, timestep, temb, rope, memory):
    """Use the normal one-pass MoT backward with IKV-derived visibility."""
    from .model import FlexAttnFunc
    layout = memory["layout"]
    seq, phase, clean = (layout[key] for key in ("seq", "phase", "clean"))
    with torch.no_grad():
        plan = build_ikv_support_plan(
            memory, hidden.device,
            return_groups=bool(memory["config"].get("compact_attention", False)))
    mot.last_sampled_ikv_capacity = memory["config"]["sampled_capacity"]
    valid = seq >= 0
    support_clean, support_noisy = plan.get("clean"), plan.get("noisy")

    def mask_mod(b, h, q_idx, kv_idx):
        same_sample = valid[q_idx] & valid[kv_idx] & (seq[q_idx] == seq[kv_idx])
        same_phase = (phase[q_idx] == phase[kv_idx]) & (clean[q_idx] == clean[kv_idx])
        past_clean = (phase[kv_idx] < phase[q_idx]) & clean[kv_idx]
        phase_row = phase[q_idx].clamp_min(0)
        retained = torch.where(clean[q_idx],
                               support_clean[phase_row, kv_idx],
                               support_noisy[phase_row, kv_idx])
        return same_sample & (same_phase | (past_clean & retained))

    length = len(seq)
    compact_groups = plan.get("groups")
    dense_mask = None
    block_mask = None
    if compact_groups is None and (hidden.device.type == "cpu" or hidden.shape[-1] < 16):
        q = torch.arange(length, device=hidden.device)[:, None]
        k = torch.arange(length, device=hidden.device)[None, :]
        dense_mask = mask_mod(None, None, q, k)[None, None]
    elif compact_groups is None:
        block_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod, 1, 1, length, length, device=hidden.device,
            BLOCK_SIZE=int(memory["config"].get("block_size", 128)), _compile=True)
        block_mask = _preserve_parent_full_blocks(block_mask)
    splits = memory["splits"]
    video = splits[0] + splits[1]
    action = splits[2] + splits[3]
    slices = [("video", 0, video), ("action", video, video + action),
              ("tactile", video + action, length)]
    mot.set_masks(self_block_mask=block_mask, dense_self_mask=dense_mask,
                  compact_self_groups=compact_groups,
                  compact_max_packed_keys=int(memory["config"].get("compact_max_packed_keys", 65536)),
                  cross_masks={})
    result = mot(hidden, text, timestep, temb, rope, slices)
    return result.masked_fill(~valid[None, :, None], 0)
