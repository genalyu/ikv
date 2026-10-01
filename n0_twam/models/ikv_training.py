"""N0-TWAM teacher-forced training with the serving IKV retention policy.

Only backbone attention execution changes. Inputs, condition corruption,
denoising targets and losses remain owned by the original Trainer/model.
No persistent inference pool or detached historical K/V is used here.
"""
import torch
import torch.distributed as dist
from .multimodal_kv_retention import make_retention_policy
from .global_kv_retention import GlobalKVRetention, RetentionConfig


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
                    if (dense is None or not dense.shape[-1]) and retention.persistence_weight:
                        raise ValueError("v2 persistence requires full-grid dense_dino_features; disable persistence_weight to omit")
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
