"""Two-rank pipeline serving with replicated IKV-v2 semantic retention."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contextlib import contextmanager
import os

import torch
import torch.distributed as dist
from torch import nn

from n0_twam import n0_twam_server as server
from models.mot import MoTBackbone, WanMoTTransformer3DModel, SharedSelfAttention
try:
    from models.mot import RetentionConfig, GlobalKVRetention
except ImportError:
    RetentionConfig = GlobalKVRetention = None
import models.utils as model_utils

# Rank 0 currently uses NumPy 2 while rank 1 uses NumPy 1. Install aliases
# only after the scientific/model stack is fully imported; doing it earlier
# interferes with binary-extension feature detection in SciPy/diffusers.
import sys
import numpy as np
try:
    import numpy.core as _numpy_core
    import numpy.core.multiarray as _numpy_multiarray
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", _numpy_multiarray)
except ImportError:
    pass


SPLIT_LAYER = int(os.environ.get("TWAM_PP_SPLIT", "15"))
CONTROL_GROUP = None


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_cpu_tree(item) for item in value]
        return type(value)(items)
    return value


def _gpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.to("cuda:0")
    if isinstance(value, dict):
        return {key: _gpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_gpu_tree(item) for item in value]
        return type(value)(items)
    return value


def _remote(op, *args, **kwargs):
    if dist.get_rank() == 0:
        dist.broadcast_object_list([{"op": op, "args": _cpu_tree(args),
                                     "kwargs": _cpu_tree(kwargs)}], src=0,
                                   group=CONTROL_GROUP, device=torch.device("cpu"))


def _owned_range(backbone):
    rank = dist.get_rank()
    return range(0, SPLIT_LAYER) if rank == 0 else range(SPLIT_LAYER, backbone.num_layers)


@torch.no_grad()
def _sync_retention_usage(policy, measurements):
    """Apply the mean query usage across *all* pipeline layers on both ranks."""
    capacity = policy.capacity
    totals = torch.zeros(2 * capacity + 1, device=policy.device, dtype=torch.float32)
    local_count = 0
    for measurement in measurements:
        if measurement is None:
            continue
        slots, mass, count = measurement
        totals[:capacity].index_add_(0, slots, mass.float())
        totals[capacity:2 * capacity].index_add_(
            0, slots, torch.full_like(mass, float(count)))
        local_count += 1
    totals[-1] = local_count
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[-1].item():
        policy.data["query_mass"].add_(totals[:capacity] / totals[-1])
        policy.data["query_exposure"].add_(totals[capacity:2 * capacity] / totals[-1])


def pp_set_masks(self, self_block_mask=None, dense_self_mask=None, cross_masks=None):
    if dist.get_rank() == 0:
        _remote("masks", self_block_mask, dense_self_mask, cross_masks)
    for layer in _owned_range(self):
        if dense_self_mask is not None:
            self.shared_attn[layer].set_dense_mask(dense_self_mask)
        else:
            self.shared_attn[layer].set_dense_mask(None)
            self.shared_attn[layer].set_block_mask(self_block_mask)
    self._cross_masks = dict(cross_masks or {})


def pp_forward(self, hidden_states, encoder_hidden_states, timestep_proj, temb,
               rotary_emb, slices, collect_cache=False, update_cache=0,
               cache_name=None, semantic_index=None, token_valid_mask=None,
               cache_metadata=None, content_observations=None):
    if collect_cache:
        raise RuntimeError("Pipeline prototype does not implement collect_cache")
    if dist.get_rank() == 0:
        _remote("forward", hidden_states, encoder_hidden_states, timestep_proj,
                temb, rotary_emb, slices, False, update_cache, cache_name,
                semantic_index, token_valid_mask, cache_metadata, content_observations)
    names = [s[0] for s in slices]
    if names != self.expert_names:
        raise ValueError(f"slice order {names} != {self.expert_names}")

    streams, mod, text, rope_e, seg_len = {}, {}, {}, {}, {}
    for name, start, end in slices:
        ex = self.experts[name]
        streams[name] = ex.embed_in(hidden_states[:, start:end])
        mod[name] = ex.modulation(
            timestep_proj[:, start:end],
            temb[:, start:end] if temb is not None else None,
        )
        text[name] = ex.text_kv(encoder_hidden_states)
        rope_e[name] = rotary_emb[:, start:end]
        seg_len[name] = end - start

    order = [name for name, _s, _e in slices]
    rank = dist.get_rank()
    peer = 1 - rank

    # IKV global-retention state is replicated per stage, while each rank owns
    # only its local layer caches. Identical metadata produces identical plans.
    ikv_mode = hasattr(self, "retention_policies")
    policy = self.retention_policies.get(cache_name) if ikv_mode else None
    cache_plan = measurements = policy_before = old_mask = rows = None
    cache_transaction = None
    owns_cache_transaction = False
    cache_entries = None
    cache_entry_start = 0
    owned = list(_owned_range(self))
    if policy is not None:
        if cache_metadata is None:
            raise ValueError("global retention requires cache token metadata")
        first = self.shared_attn[owned[0]].attn_caches[cache_name]
        for layer in owned[1:]:
            other = self.shared_attn[layer].attn_caches[cache_name]
            if (not torch.equal(first["mask"], other["mask"]) or
                    not torch.equal(first["id"], other["id"])):
                raise RuntimeError("local pipeline-stage cache slots diverged")
        _, validity = SharedSelfAttention._normalise_semantic_sequence(
            semantic_index, hidden_states.shape[0], hidden_states.shape[1],
            hidden_states.device, token_valid_mask=token_valid_mask)
        positions = (torch.arange(hidden_states.shape[1], device=hidden_states.device)
                     if validity is None else validity[0].nonzero().flatten())
        rows = {name: value[positions] for name, value in cache_metadata.items()}
        old_mask = first["mask"].clone()
        policy_before = policy.snapshot() if update_cache else None
        cache_plan = policy.plan(old_mask, len(positions), rows)
        measurements = []
    if ikv_mode and update_cache != 0 and cache_name is not None:
        cache_transaction = self._active_cache_transactions.get(cache_name)
        if cache_transaction is None:
            cache_transaction = self.begin_cache_transaction(cache_name)
            owns_cache_transaction = True
        cache_entries = cache_transaction["entries"]
        cache_entry_start = len(cache_entries)

    if rank == 1:
        for idx, name in enumerate(order):
            incoming = torch.empty_like(streams[name])
            dist.recv(incoming, src=peer, tag=100 + idx)
            streams[name] = incoming

    try:
        for layer in owned:
            q_chunks, k_chunks, v_chunks, post = [], [], [], []
            for name in order:
                block = self.experts[name].blocks[layer]
                q, k, v, residual, gate, cs, csc, cg = block(
                    streams[name], temb=mod[name], rotary_emb=rope_e[name], mot_mode="pre"
                )
                q_chunks.append(q); k_chunks.append(k); v_chunks.append(v)
                post.append((name, block, residual, gate, cs, csc, cg))
            q = torch.cat(q_chunks, dim=1)
            k = torch.cat(k_chunks, dim=1)
            v = torch.cat(v_chunks, dim=1)
            attn_kwargs = dict(update_cache=update_cache, cache_name=cache_name)
            if ikv_mode:
                attn_kwargs.update(
                    semantic_index=semantic_index,
                    token_valid_mask=token_valid_mask,
                    cache_transaction=cache_entries,
                    cache_plan=cache_plan,
                    usage_collector=(None if policy is None or not any((
                        policy.config.query_weight, policy.config.action_query_weight,
                        policy.config.tactile_query_weight)) else (policy, measurements)),
                    cache_observation_flags=(None if cache_metadata is None
                                             else cache_metadata["observation_flag"]),
                    manage_semantic_sidecar=(layer == owned[0]),
                )
            attn = self.shared_attn[layer](q, k, v, **attn_kwargs)
            if ikv_mode and layer == owned[0] and update_cache and cache_name is not None:
                self._share_semantic_sidecar(cache_name)
            cursor = 0
            updated = {}
            for name, block, residual, gate, cs, csc, cg in post:
                sl = attn[:, cursor:cursor + seg_len[name]]
                do_cross = self.experts[name].do_cross_attn and text[name] is not None
                updated[name] = block(
                    residual, mot_mode="post",
                    mot_kwargs=dict(
                        attn_output=sl, gate_msa=gate, c_shift_msa=cs,
                        c_scale_msa=csc, c_gate_msa=cg,
                        encoder_hidden_states=text[name], do_cross_attn=do_cross,
                        cross_attn_mask=self._cross_masks.get(name),
                    ),
                )
                cursor += seg_len[name]
            streams = updated
    except BaseException:
        if policy_before is not None:
            policy.restore(policy_before)
        if cache_transaction is not None:
            if owns_cache_transaction:
                self.rollback_cache_transaction(cache_transaction)
            else:
                self._rollback_cache_entries(cache_entries, start=cache_entry_start)
        raise
    if rank == 0:
        for idx, name in enumerate(order):
            dist.send(streams[name].contiguous(), dst=peer, tag=100 + idx)
        for idx, name in enumerate(order):
            incoming = torch.empty_like(streams[name])
            dist.recv(incoming, src=peer, tag=200 + idx)
            streams[name] = incoming
    else:
        for idx, name in enumerate(order):
            dist.send(streams[name].contiguous(), dst=peer, tag=200 + idx)

    # Wait until the activation exchange finishes: otherwise rank 0 would
    # block in all-reduce while rank 1 is still waiting for those activations.
    try:
        if policy is not None and update_cache:
            policy.commit(cache_plan[0], rows, old_mask)
            _sync_retention_usage(policy, measurements)
            if content_observations is not None and policy.config.version == 2:
                policy.observe_dense(*content_observations)
        if owns_cache_transaction:
            self.commit_cache_transaction(cache_transaction)
    except BaseException:
        if policy_before is not None:
            policy.restore(policy_before)
        if cache_transaction is not None:
            if owns_cache_transaction:
                self.rollback_cache_transaction(cache_transaction)
            else:
                self._rollback_cache_entries(cache_entries, start=cache_entry_start)
        raise

    if rank == 1:
        return None
    return torch.cat([self.experts[name].embed_out(streams[name])
                      for name, _s, _e in slices], dim=1)


def pp_share_semantic_sidecar(self, cache_name):
    owned = list(_owned_range(self))
    first = self.shared_attn[owned[0]].attn_caches.get(cache_name)
    if first is not None:
        for layer in owned[1:]:
            cache = self.shared_attn[layer].attn_caches.get(cache_name)
            if cache is not None:
                cache['semantic'] = first.get('semantic')


MoTBackbone._share_semantic_sidecar = pp_share_semantic_sidecar
MoTBackbone.set_masks = pp_set_masks
MoTBackbone.forward = pp_forward


def pp_clear_cache(self, cache_name):
    if dist.get_rank() == 0:
        _remote("clear_cache", cache_name)
    for layer in _owned_range(self.mot):
        self.mot.shared_attn[layer].clear_cache(cache_name)
    self.mot.retention_policies.pop(cache_name, None)
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()  # Return inactive allocator blocks before NeoSim resets.
    print(f"[pp_reset] rank={dist.get_rank()} released_mib="
          f"{(reserved_before - torch.cuda.memory_reserved()) // 1048576}", flush=True)


def pp_clear_pred_cache(self, cache_name, *, include_observed=False):
    if dist.get_rank() == 0:
        _remote("clear_pred_cache", cache_name, include_observed=include_observed)
    owned = list(_owned_range(self.mot))
    entries = getattr(self.mot, '_active_cache_transactions', {}).get(
        cache_name, {}).get('entries')
    if entries is not None:
        for layer in owned:
            attention = self.mot.shared_attn[layer]
            cache = attention.attn_caches.get(cache_name)
            if cache is not None:
                selected = cache['mask'] if include_observed else cache['mask'] & cache['is_pred']
                slots = selected.nonzero().flatten()
                if slots.numel():
                    entries.append((attention, cache_name,
                                    attention._snapshot_cache_slots(cache, slots)))
    for layer in owned:
        self.mot.shared_attn[layer].clear_pred_cache(cache_name, include_observed=include_observed)


def pp_create_empty_cache(self, cache_name, attn_window,
                          latent_token_per_chunk, action_token_per_chunk,
                          device, dtype, batch_size):
    if dist.get_rank() == 0:
        _remote("create_cache", cache_name, attn_window,
                latent_token_per_chunk, action_token_per_chunk,
                device, dtype, batch_size)
    total_tokens = (attn_window // 2) * latent_token_per_chunk + (
        attn_window // 2) * action_token_per_chunk
    for layer in _owned_range(self.mot):
        self.mot.shared_attn[layer].init_kv_cache(
            cache_name, total_tokens, self.num_attention_heads,
            self.attention_head_dim, device, dtype, batch_size)
    self.mot.retention_policies.pop(cache_name, None)


WanMoTTransformer3DModel.clear_cache = pp_clear_cache
WanMoTTransformer3DModel.clear_pred_cache = pp_clear_pred_cache
WanMoTTransformer3DModel.create_empty_cache = pp_create_empty_cache


# Grounding/inference transactions must bracket BOTH pipeline stages.
_original_cache_transaction = getattr(WanMoTTransformer3DModel, 'cache_transaction', None)


@contextmanager
def pp_cache_transaction(self, cache_name):
    _remote('begin_transaction', cache_name)
    try:
        with _original_cache_transaction(self, cache_name) as transaction:
            yield transaction
    except BaseException:
        _remote('rollback_transaction')
        raise
    else:
        _remote('commit_transaction')


if _original_cache_transaction is not None:
    WanMoTTransformer3DModel.cache_transaction = pp_cache_transaction


def pp_configure_global_retention(self, cache_name, **config):
    if dist.get_rank() == 0:
        _remote("configure_retention", cache_name, **config)
    if self.use_rgb_motion_tokens and config.get("version", 1) != 2:
        raise ValueError("sparse global retention requires version 2")
    owned = list(_owned_range(self.mot))
    caches = [self.mot.shared_attn[i].attn_caches[cache_name] for i in owned]
    if not caches or any(c["mask"].any() for c in caches):
        raise ValueError("configure retention on a new, empty local-stage KV cache")
    cfg = RetentionConfig(**config)
    from n0_twam.models.multimodal_kv_retention import make_retention_policy
    self.mot.retention_policies[cache_name] = make_retention_policy(
        caches[0]["mask"].numel(), caches[0]["mask"].device, cfg)


def _pp_retention_mask(self, cache_name):
    layer = next(iter(_owned_range(self.mot)))
    return self.mot.shared_attn[layer].attn_caches[cache_name]["mask"]


def pp_get_global_retention(self, cache_name):
    policy = self.mot.retention_policies.get(cache_name)
    if policy is None:
        return None
    mask = _pp_retention_mask(self, cache_name)
    slots = mask.nonzero().flatten()
    return {"t0": policy.t0, "slot_indices": slots.clone(),
            **{k: v[slots].detach().clone() for k, v in policy.data.items()},
            "score": policy.scores(slots).detach().clone(),
            "components": {k: v.detach().clone()
                           for k, v in policy.components(slots).items()}}


def pp_video_index_handle(self, cache_name, since_uid):
    if dist.get_rank() == 0:
        _remote("video_handle", cache_name, since_uid)
    policy = self.mot.retention_policies[cache_name]
    return policy.video_handle(_pp_retention_mask(self, cache_name), since_uid)


def pp_annotate_video_dino(self, cache_name, handle, features):
    if dist.get_rank() == 0:
        _remote("annotate_dino", cache_name, features)
    policy = self.mot.retention_policies[cache_name]
    if handle["owner"] is not policy:
        raise ValueError("video index handle belongs to a different cache generation")
    mask = None
    for layer in _owned_range(self.mot):
        mask = self.mot.shared_attn[layer].attn_caches[cache_name]["mask"]
        if not mask[handle["slots"]].all():
            raise ValueError("video index handle addresses removed local-stage KV")
    return policy.annotate_video_dino(mask, handle, features)


WanMoTTransformer3DModel.configure_global_retention = pp_configure_global_retention
WanMoTTransformer3DModel.get_global_retention = pp_get_global_retention
WanMoTTransformer3DModel.video_index_handle = pp_video_index_handle
WanMoTTransformer3DModel.annotate_video_dino = pp_annotate_video_dino

_load_mot = model_utils.load_mot_checkpoint


def load_mot_cpu(*args, **kwargs):
    kwargs["torch_device"] = "cpu"
    return _load_mot(*args, **kwargs)


model_utils.load_mot_checkpoint = load_mot_cpu


def pipeline_place(model):
    global CONTROL_GROUP
    CONTROL_GROUP = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 2 or not 0 < SPLIT_LAYER < model.mot.num_layers:
        raise ValueError("PP requires two ranks and a split inside the layer range")
    for name in model.mot.expert_names:
        blocks = model.mot.experts[name].blocks
        for layer in range(model.mot.num_layers):
            keep = layer < SPLIT_LAYER if rank == 0 else layer >= SPLIT_LAYER
            if not keep:
                blocks[layer] = nn.Identity()
    for layer in range(model.mot.num_layers):
        keep = layer < SPLIT_LAYER if rank == 0 else layer >= SPLIT_LAYER
        if not keep:
            model.mot.shared_attn[layer] = nn.Identity()
    model.eval().requires_grad_(False)
    return model.to(device=torch.device("cuda:0"), dtype=torch.bfloat16)


server.shard_model = pipeline_place

from utils import server_utils as _server_utils


class _PipelinePolicy:
    def __init__(self, model):
        self.model = model

    def infer(self, obs):
        return self.model.infer(obs)


def _pipeline_worker(model):
    transformer = model.transformer
    last_video_handle = None
    active_transaction = None
    while True:
        packet = [None]
        dist.broadcast_object_list(packet, src=0, group=CONTROL_GROUP,
                                   device=torch.device("cpu"))
        command = packet[0]
        op = command["op"]
        args = _gpu_tree(command["args"])
        kwargs = _gpu_tree(command["kwargs"])
        if op == "stop":
            break
        if op == 'begin_transaction':
            if active_transaction is not None:
                raise RuntimeError('nested remote cache transaction')
            active_transaction = transformer.mot.begin_cache_transaction(*args)
        elif op == 'commit_transaction':
            transformer.mot.commit_cache_transaction(active_transaction)
            active_transaction = None
        elif op == 'rollback_transaction':
            transformer.mot.rollback_cache_transaction(active_transaction)
            active_transaction = None
        elif op == "clear_cache":
            transformer.clear_cache(*args)
            last_video_handle = None
        elif op == "clear_pred_cache":
            transformer.clear_pred_cache(*args, **kwargs)
        elif op == "create_cache":
            transformer.create_empty_cache(*args)
        elif op == "configure_retention":
            transformer.configure_global_retention(*args, **kwargs)
        elif op == "video_handle":
            last_video_handle = transformer.video_index_handle(*args)
        elif op == "annotate_dino":
            if last_video_handle is None:
                raise RuntimeError("rank 1 has no matching video index handle")
            transformer.annotate_video_dino(args[0], last_video_handle, args[1])
            last_video_handle = None
        elif op == "masks":
            transformer.mot.set_masks(*args)
        elif op == "forward":
            transformer.mot.forward(*args)
        else:
            raise RuntimeError(f"unknown pipeline command: {op}")


def _pipeline_serve(model, local_rank, host, port):
    if dist.get_rank() == 0:
        policy = _PipelinePolicy(model)
        _server_utils.WebsocketPolicyServer(policy, host=host, port=port).serve_forever()
        _remote("stop")
    else:
        _pipeline_worker(model)


server.run_async_server_mode = _pipeline_serve

_server_init = server.TWAM_Server.__init__
_encode_prompt = server.TWAM_Server.encode_prompt
_prompt_cache = {}


def cached_encode_prompt(self, *args, **kwargs):
    prompt = kwargs.get("prompt", args[0] if args else None)
    key = (
        str(prompt), str(kwargs.get("negative_prompt")),
        bool(kwargs.get("do_classifier_free_guidance", True)),
        int(kwargs.get("num_videos_per_prompt", 1)),
    )
    if key not in _prompt_cache:
        _prompt_cache[key] = _encode_prompt(self, *args, **kwargs)
    return _prompt_cache[key]


server.TWAM_Server.encode_prompt = cached_encode_prompt


def init_with_gpu_vae(self, cfg):
    _server_init(self, cfg)
    if dist.get_rank() == 0:
        self.vae.to(device=self.device, dtype=self.dtype)


server.TWAM_Server.__init__ = init_with_gpu_vae


def main():
    import json
    cfg = server.TWAM_CONFIGS["multitask_server"]
    cfg.host = "0.0.0.0"
    cfg.enable_offload = True
    overrides = os.environ.get("IKV_SERVE_OVERRIDES")
    if not overrides:
        raise ValueError("IKV_SERVE_OVERRIDES must point to exported semantic serving config")
    with open(overrides) as stream:
        cfg.update(json.load(stream))
    cfg.kv_retention = dict(cfg.kv_retention)
    if cfg.kv_retention.get("version", 1) != 2:
        raise ValueError("semantic PP requires an IKV-v2 trained checkpoint")
    cfg.kv_index_dino_device = os.environ.get("IKV_DINO_DEVICE", "cpu")
    server.init_logger()
    server.run(argparse.Namespace(
        config_name="multitask_server",
        port=int(os.environ.get("TWAM_WS_PORT", "29960")),
        save_root=os.environ["TWAM_SERVE_OUT"],
    ))


if __name__ == "__main__":
    main()
