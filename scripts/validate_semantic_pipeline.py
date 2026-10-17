"""Compare two-host pipeline retention/rollback against a full two-layer MoT.

Runs on CPU/Gloo, with tiny random policy weights. Exercises the production PP
functions without importing or starting the heavyweight inference service.
"""
import argparse
import ast
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sys
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import torch
import torch.distributed as dist
from contextlib import contextmanager
from n0_twam.models.mot import SharedSelfAttention
from n0_twam.models.global_kv_retention import RetentionConfig
from test_global_kv_retention import tiny_model, model_input


def production_functions():
    source = Path(__file__).with_name("semantic_pipeline.py")
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    namespace = dict(torch=torch, dist=dist, SharedSelfAttention=SharedSelfAttention,
                     RetentionConfig=RetentionConfig, contextmanager=contextmanager,
                     SPLIT_LAYER=1, CONTROL_GROUP=dist.group.WORLD)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    # Same wire format and production methods, CPU tensors for this small check.
    namespace["_gpu_tree"] = namespace["_cpu_tree"]
    return namespace


def wire(model, namespace):
    for method, implementation in {
        "set_masks": "pp_set_masks", "forward": "pp_forward",
        "_share_semantic_sidecar": "pp_share_semantic_sidecar"
    }.items():
        setattr(model.mot, method, MethodType(namespace[implementation], model.mot))
    for method, implementation in {
        "create_empty_cache": "pp_create_empty_cache", "clear_cache": "pp_clear_cache",
        "clear_pred_cache": "pp_clear_pred_cache", "configure_global_retention": "pp_configure_global_retention",
        "get_global_retention": "pp_get_global_retention", "cache_transaction": "pp_cache_transaction",
        "video_index_handle": "pp_video_index_handle", "annotate_video_dino": "pp_annotate_video_dino"
    }.items():
        setattr(model, method, MethodType(namespace[implementation], model))


def state(model, layer):
    cache = model.mot.shared_attn[layer].attn_caches["test"]
    names = ("k", "v", "id", "mask", "is_pred", "semantic", "predicted_kv", "predicted_kinds")
    policy = model.mot.retention_policies.get("test")
    cache_values = None if cache is None else deepcopy({name: cache.get(name) for name in names})
    if cache_values is not None:
        for name in ("k", "v"):
            cache_values[name] = cache_values[name][:, cache["mask"]]
    snapshot = None if policy is None else policy.snapshot()
    if policy is not None and getattr(policy, "archive", None) is not None:
        # KV byte totals depend on the number of locally owned layers. Compare
        # immutable token metadata, plus this stage's actual layer KV below.
        records, stats, files = snapshot[2]
        snapshot = snapshot[:2] + ([record["metadata"] for record in records],) + snapshot[3:]
        cache_values["archived_layer_kv"] = [record["kv"][layer] for record in records]
    return {"cache": cache_values,
            "retention": model.get_global_retention("test"),
            "policy_snapshot": snapshot}


def equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected: equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected): equal(a, e)
    else:
        assert actual == expected, (actual, expected)


def worker(model, namespace):
    transaction = None
    while True:
        packet = [None]
        dist.broadcast_object_list(packet, src=0)
        command = packet[0]
        op, args, kwargs = command["op"], command["args"], command["kwargs"]
        if op == "stop": return
        if op == "inspect":
            dist.send_object_list([state(model, 1)], dst=0)
        elif op == "begin_transaction":
            transaction = model.mot.begin_cache_transaction(*args)
        elif op == "commit_transaction":
            model.mot.commit_cache_transaction(transaction); transaction = None
        elif op == "rollback_transaction":
            model.mot.rollback_cache_transaction(transaction); transaction = None
        elif op in ("forward", "masks"):
            with torch.no_grad():
                getattr(model.mot, {"masks": "set_masks"}.get(op, op))(*args, **kwargs)
        else:
            names = {"create_cache": "create_empty_cache", "configure_retention": "configure_global_retention",
                     "clear_cache": "clear_cache", "clear_pred_cache": "clear_pred_cache"}
            getattr(model, names[op])(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu-archive", action="store_true")
    parser.add_argument("--sys2-mock", action="store_true", help="deterministic scoring transport, real PP/cache logic")
    parser.add_argument("--ikv-interval-frames", type=int, default=1)
    args = parser.parse_args()
    if args.sys2_mock:
        from n0_twam.preprocessing.sys2 import Sys2Client
        Sys2Client.score = lambda self,candidates,current_time: {
            c["id"]: 100 if int(c["id"]) % 3 == 0 else 10 for c in candidates}
        args.cpu_archive = True
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=args.address, rank=args.rank,
                            world_size=2, timeout=timedelta(seconds=90))
    torch.manual_seed(1729)
    baseline = tiny_model(False)
    stage = deepcopy(baseline)
    namespace = production_functions()
    namespace["_original_cache_transaction"] = type(baseline).cache_transaction
    wire(stage, namespace)
    if args.rank == 1:
        worker(stage, namespace)
        dist.destroy_process_group()
        return
    config = dict(version=2, ikv_interval_frames=args.ikv_interval_frames, video_capacity=8, action_capacity=8, tactile_capacity=8,
                  task_weight=1., class_recency_weight=1., visual_weight=0.,
                  persistence_weight=0., query_weight=0., action_query_weight=0., tactile_query_weight=0.)
    if args.cpu_archive:
        config.update(history_storage="cpu_archive", archive_max_bytes=1024**2)
    if args.sys2_mock:
        config.update(score_backend="qwen_sys2",sys2_endpoint="http://localhost:1",sys2_task="test")
    def initialize(model):
        model.create_empty_cache("test", 2, 16, 8, torch.device("cpu"), torch.float32, 1)
        model.configure_global_retention("test", **config)
    def inspect():
        namespace["_remote"]("inspect")
        packet = [None]; dist.recv_object_list(packet, src=1)
        equal(state(stage, 0), state(baseline, 0))
        equal(packet[0], state(baseline, 1))
    def forward(time, *, prediction=False, action=False):
        data = model_input(time, action=action)
        if not action:
            data["kv_index"] = dict(dino=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]),
                task_relevance=torch.tensor([.9,.1,.8,.2]), observation_flag=int(not prediction))
        if args.sys2_mock:
            index = dict(sys2_frame=torch.full((4,),time+1,dtype=torch.long),
                         sys2_patch=torch.arange(4),sys2_height=torch.full((4,),2,dtype=torch.long),
                         sys2_width=torch.full((4,),2,dtype=torch.long),observation_flag=int(not prediction))
            if not action: data["kv_index"] = index
            data["tactile_kv_index"] = dict(index,contact_present=True)
        mode = 1 if prediction else 2
        with torch.no_grad():
            expected = baseline(deepcopy(data), action_mode=action, update_cache=mode, cache_name="test")
            actual = stage(deepcopy(data), action_mode=action, update_cache=mode, cache_name="test")
        equal(actual, expected)
        inspect()
    initialize(baseline); initialize(stage)
    for time in range(4): forward(time)
    forward(4, prediction=True)
    baseline.clear_pred_cache("test"); stage.clear_pred_cache("test"); inspect()
    forward(5, action=True)
    before = state(stage, 0)
    try:
        with baseline.cache_transaction("test"), stage.cache_transaction("test"):
            forward(6)
            raise RuntimeError("intentional rollback")
    except RuntimeError as exc:
        assert str(exc) == "intentional rollback"
    equal(state(stage, 0), before); inspect()
    with baseline.cache_transaction("test"), stage.cache_transaction("test"): forward(7)
    inspect()
    baseline.clear_cache("test"); stage.clear_cache("test")
    assert not stage.mot.retention_policies
    initialize(baseline); initialize(stage)
    forward(0)
    report = dict(status="passed", sys2_mock=args.sys2_mock, ranks=2, layers=2, backend="gloo", device="cpu",
                  history_storage="cpu_archive" if args.cpu_archive else "evict",
                  checks=["observed video", "score eviction", "predicted video", "clear predictions",
                          "action stream", "transaction rollback", "transaction commit", "episode reset",
                          "per-layer KV and both-rank score/history parity"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    namespace["_remote"]("stop")
    dist.destroy_process_group()


if __name__ == "__main__": main()
