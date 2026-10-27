"""Create a pinned native serve bundle from a Memory checkpoint manifest."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from .selection import VERSION
from .sidecar import verify_binding
from .serving import install


def prepare(spec):
    checkpoint=Path(spec["checkpoint"])
    manifest=json.loads((checkpoint/"memory_manifest.json").read_text())
    if manifest["version"]!=VERSION or manifest["chunk_size"]!=2 or not manifest["rgb_only"]:
        raise ValueError("Checkpoint is not a compatible RGB Memory run")
    if manifest["experiment"]["model"]!=spec["model"]:
        raise ValueError("Foundation model differs from checkpoint")
    if not (checkpoint/"training_state/complete.json").is_file():
        raise ValueError("Checkpoint save is incomplete")
    bundle=Path(spec["bundle"]);bundle.mkdir(parents=True,exist_ok=True)
    sources=dict(transformer=checkpoint/"transformer")
    sources.update({k:Path(spec["base_model"])/k for k in ("vae","tokenizer","text_encoder")})
    for key,target in sources.items():
        if not target.exists():raise FileNotFoundError(target)
        path=bundle/key
        if path.exists():
            if not path.samefile(target):raise ValueError("Bundle points at a different model")
        else:path.symlink_to(target.absolute(),target_is_directory=True)
    if spec["model"]=="n0twam":
        from n0_twam.n0_twam_server import TWAM_Server
        from n0_twam.configs import TWAM_CONFIGS
        import n0_twam.n0_twam_server as module
        Server=TWAM_Server;cfg=copy.deepcopy(TWAM_CONFIGS["twam_server"])
        meta=json.loads((checkpoint/"train_meta.json").read_text())
        cfg.update({k:v for k,v in meta.items() if k!="step"})
        cfg.kv_index_dino_online=False;cfg.kv_index_predicted_dino=False
        cfg.kv_neoforce_online=False;cfg.kv_semantic_encoder={}
    elif spec["model"]=="lingbot":
        directory=Path(spec["source_root"])/"wan_va"
        sys.path.insert(0,str(directory))
        loader=importlib.util.spec_from_file_location("memory_native_lingbot_server",directory/"wan_va_server.py")
        module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
        Server=module.VA_Server;cfg=copy.deepcopy(module.VA_CONFIGS["robotwin"])
        cfg.used_action_channel_ids=manifest["normalization"]["used_action_channel_ids"]
        cfg.action_norm_method="quantiles"
    else:raise ValueError("Unknown foundation model")
    cfg.update(wan22_pretrained_model_name_or_path=str(bundle),save_root=spec["save_root"],
        height=256,width=256,env_type="none",frame_chunk_size=2,
        action_dim=manifest["action_dimension"],action_per_frame=manifest["action_per_frame"],
        obs_cam_keys=manifest["cameras"],norm_stat=manifest["normalization"],
        tactile_keys=[],use_local_tactile=False,tactile_global_zero=True,use_rgb_motion_tokens=False,
        attn_mode="torch",cold_seed_mode="free",infer_mode="server",host=spec.get("host","127.0.0.1"),
        port=int(spec.get("port",29611)),enable_offload=True)
    inference=spec.get("inference",{})
    allowed={"guidance_scale","action_guidance_scale","num_inference_steps",
        "action_num_inference_steps","video_exec_step","snr_shift","action_snr_shift","attn_window"}
    if set(inference)-allowed:raise ValueError("Unsupported inference override")
    for key,value in inference.items():
        if key in {"num_inference_steps","action_num_inference_steps","attn_window"}:
            if type(value) is not int or value<1:raise ValueError("Positive integer inference setting required")
        elif key=="video_exec_step":
            if type(value) is not int or value==0 or value < -1:raise ValueError("Invalid video execution steps")
        elif not isinstance(value,(int,float)) or not __import__("math").isfinite(value) or value<=0:
            raise ValueError("Positive finite inference setting required")
    cfg.update(inference)
    inverse=[len(cfg.used_action_channel_ids)]*cfg.action_dim
    for i,j in enumerate(cfg.used_action_channel_ids):inverse[j]=i
    cfg.inverse_used_action_channel_ids=inverse
    if manifest["mode"]=="scorer":
        binding=verify_binding(dict(binding=spec["binding"]))
        if binding["scorer_sha256"]!=manifest["scorer_sha256"]:
            raise ValueError("Serving scorer differs from the one used in training")
        cfg.memory_scorer_sha256=binding["scorer_sha256"]
        cfg.memory_scorer_endpoint=spec["scorer_endpoint"]
        cfg.kv_cache_policy="global";cfg.use_ikv_training=True;cfg.ikv_train_capacity=4096
        if spec["model"]=="lingbot":
            cfg.kv_retention=dict(version=2,video_capacity=3584,action_capacity=512,tactile_capacity=0,
                query_weight=0.,action_query_weight=0.,tactile_query_weight=0.,visual_weight=0.,
                time_weight=0.,contact_weight=0.,persistence_weight=0.,class_recency_weight=0.,task_weight=0.)
        # N0's metadata cross-check also covers legacy retention fields. Keep
        # that exact envelope; install() replaces the policy with the bound
        # frozen scorer before any token can enter a newly reset cache.
        Server=type("MemoryNativeServer",(Server,),{})
        install(Server,spec["model"])
    return module,Server,cfg,manifest


def configuration_contract(cfg,manifest):
    import hashlib
    keys=["frame_chunk_size","action_dim","action_per_frame","used_action_channel_ids","obs_cam_keys",
        "cold_seed_mode","guidance_scale","action_guidance_scale","num_inference_steps",
        "action_num_inference_steps","video_exec_step","snr_shift","action_snr_shift","attn_window"]
    result={k:getattr(cfg,k,None) for k in keys}
    result.update(version=VERSION,checkpoint_step=manifest["step"],mode=manifest["mode"],
        scorer_sha256=manifest["scorer_sha256"],normalization_sha256=hashlib.sha256(
            json.dumps(manifest["normalization"],sort_keys=True).encode()).hexdigest())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--check-only",action="store_true")
    args=p.parse_args();spec=json.loads(Path(args.config).read_text())
    module,Server,cfg,manifest=prepare(spec)
    if args.check_only:
        print(json.dumps(dict(status="configuration_validated",model=spec["model"],mode=manifest["mode"],
            cameras=cfg.obs_cam_keys,action_dimension=cfg.action_dim,used_action_channel_ids=cfg.used_action_channel_ids,
            chunk_size=cfg.frame_chunk_size)));return
    rank=int(os.getenv("RANK",0));local=int(os.getenv("LOCAL_RANK",0));world=int(os.getenv("WORLD_SIZE",1))
    module.init_distributed(world,local,rank);cfg.rank=rank;cfg.local_rank=local;cfg.world_size=world
    module.init_logger()
    server=Server(cfg)
    Path(cfg.save_root).mkdir(parents=True,exist_ok=True)
    (Path(cfg.save_root)/"serve_contract.json").write_text(json.dumps(configuration_contract(cfg,manifest),indent=2))
    module.run_async_server_mode(server,local,cfg.host,cfg.port)

if __name__=="__main__":main()
