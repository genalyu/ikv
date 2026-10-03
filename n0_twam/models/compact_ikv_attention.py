"""Exact retained-key attention groups for one-pass masked IKV training.

Each query group contains one phase and one clean/noisy branch. Its keys are
exactly the same-phase branch plus the historical clean tokens selected by the
serving retention policy for that query phase. No token-level attention mask
is materialized, and index_select/index_copy keep all Q/K/V gradients intact.
"""
import torch

from .model import custom_sdpa


@torch.no_grad()
def build_compact_ikv_groups(layout, support):
    seq, phase, clean = (layout[name] for name in ("seq", "phase", "clean"))
    valid = seq >= 0
    if torch.any(valid & (seq != 0)):
        raise ValueError("Compact IKV currently requires per-rank batch_size=1")
    groups = []
    covered = []
    for stage in torch.unique(phase[valid], sorted=True).tolist():
        same = valid & (phase == stage)
        earlier_clean = valid & clean & (phase < stage)
        for branch in (True, False):
            query = (same & (clean == branch)).nonzero().flatten()
            if not query.numel():
                continue
            retained = support["clean" if branch else "noisy"][int(stage)]
            history = (earlier_clean & retained).nonzero().flatten()
            keys = torch.cat((history, query)).sort().values
            groups.append((query, keys))
            covered.append(query)
    if not groups or torch.cat(covered).numel() != valid.sum().item():
        raise ValueError("Compact IKV query groups do not cover valid tokens")
    return groups


def compact_ikv_attention(q, k, v, groups):
    """Compute only visible Q-KV rectangles, with full differentiable gathers."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("Expected matching Q/K/V shaped [B,S,H,D]")
    if not groups:
        return torch.zeros_like(q)
    outputs = []
    indices = []
    for query, keys in groups:
        outputs.append(custom_sdpa(
            q.index_select(1, query),
            k.index_select(1, keys),
            v.index_select(1, keys),
        ))
        indices.append(query)
    return torch.zeros_like(q).index_copy(
        1, torch.cat(indices), torch.cat(outputs, dim=1))


def packed_compact_ikv_attention(q, k, v, groups):
    """One varlen FlashAttention call for all exact-visibility groups.

    This remains experimental: duplicated gathered historical KV can consume
    considerably more memory than the block-sparse path on long sequences.
    """
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Expected matching Q/K/V shaped [1,S,H,D]")
    if not groups:
        return torch.zeros_like(q)
    query_indices = torch.cat([row[0] for row in groups])
    key_indices = torch.cat([row[1] for row in groups])
    q_lengths = [int(row[0].numel()) for row in groups]
    k_lengths = [int(row[1].numel()) for row in groups]
    q_offsets = torch.tensor([0] + list(torch.tensor(q_lengths).cumsum(0).tolist()),
                             device=q.device, dtype=torch.int32)
    k_offsets = torch.tensor([0] + list(torch.tensor(k_lengths).cumsum(0).tolist()),
                             device=q.device, dtype=torch.int32)
    gathered_q = q[0].index_select(0, query_indices)
    gathered_k = k[0].index_select(0, key_indices)
    gathered_v = v[0].index_select(0, key_indices)
    output = torch.ops.aten._flash_attention_forward(
        gathered_q, gathered_k, gathered_v, q_offsets, k_offsets,
        max(q_lengths), max(k_lengths), 0.0, False, False,
    )[0]
    return torch.zeros_like(q).index_copy(
        1, query_indices, output.unsqueeze(0))


@torch.no_grad()
def _pack_group_chunks(groups, device, max_keys):
    chunks = []
    current = []
    key_count = 0

    def add_chunk(rows):
        qi = torch.cat([row[0] for row in rows])
        ki = torch.cat([row[1] for row in rows])
        q_lengths = [int(row[0].numel()) for row in rows]
        k_lengths = [int(row[1].numel()) for row in rows]
        q_offsets = torch.tensor(
            [0] + list(torch.tensor(q_lengths).cumsum(0).tolist()),
            device=device, dtype=torch.int32)
        k_offsets = torch.tensor(
            [0] + list(torch.tensor(k_lengths).cumsum(0).tolist()),
            device=device, dtype=torch.int32)
        chunks.append((qi, ki, q_offsets, k_offsets,
                       max(q_lengths), max(k_lengths)))

    for row in groups:
        size = int(row[1].numel())
        if current and key_count + size > max_keys:
            add_chunk(current)
            current = []
            key_count = 0
        current.append(row)
        key_count += size
    if current:
        add_chunk(current)
    return chunks


def _flash_varlen(q, k, v, chunk):
    qi, ki, cq, ck, max_q, max_k = chunk
    packed_q = q[0].index_select(0, qi)
    packed_k = k[0].index_select(0, ki)
    packed_v = v[0].index_select(0, ki)
    y, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
        packed_q, packed_k, packed_v, cq, ck,
        max_q, max_k, 0.0, False, False)
    return y, lse, rng, unused


class _StreamedPackedIKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, chunks):
        ctx.chunks = chunks
        output = torch.zeros_like(q)
        lses, rngs, unuseds = [], [], []
        for chunk in chunks:
            y, lse, rng, unused = _flash_varlen(q, k, v, chunk)
            output[0].index_copy_(0, chunk[0], y)
            lses.append(lse)
            rngs.append(rng)
            unuseds.append(unused)
        # FlashAttention backward accepts these exact forward artifacts. Keep
        # only the output and small metadata; gather Q/K/V again as before.
        ctx.save_for_backward(q, k, v, output, *lses, *rngs, *unuseds)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        q, k, v, output = saved[:4]
        count = len(ctx.chunks)
        lses = saved[4:4 + count]
        rngs = saved[4 + count:4 + 2 * count]
        unuseds = saved[4 + 2 * count:4 + 3 * count]
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        for index, chunk in enumerate(ctx.chunks):
            qi, ki = chunk[:2]
            packed_q = q[0].index_select(0, qi)
            packed_k = k[0].index_select(0, ki)
            packed_v = v[0].index_select(0, ki)
            gq, gk, gv = torch.ops.aten._flash_attention_backward(
                grad_output[0].index_select(0, qi), packed_q, packed_k, packed_v,
                output[0].index_select(0, qi), lses[index],
                chunk[2], chunk[3], chunk[4], chunk[5],
                0.0, False, rngs[index], unuseds[index])
            dq[0].index_copy_(0, qi, gq)
            dk[0].index_add_(0, ki, gk.float())
            dv[0].index_add_(0, ki, gv.float())
        return dq, dk.to(k.dtype), dv.to(v.dtype), None


def streamed_packed_ikv_attention(q, k, v, groups, max_keys=65536, *, chunks=None):
    """Bound gather memory; recompute each packed chunk during backward."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Expected matching Q/K/V shaped [1,S,H,D]")
    if not groups:
        return torch.zeros_like(q)
    if chunks is None:
        chunks = _pack_group_chunks(groups, q.device, max_keys)
    return _StreamedPackedIKV.apply(q, k, v, chunks)
