"""Group-relative native-KV selection with the existing frozen N0 FM evaluator.
Manifest: feature_path plus n0_batch, target_raw_frame, query_frame,
alignment_provenance, video_candidate_map and optional tactile_candidate_map.
Each map covers ALL clean native tokens (including target); values are feature
row indices, or -1 ONLY for the final target tokens. Mapping is explicit.
"""
import argparse,json,random
from pathlib import Path
from copy import deepcopy
import torch
from .model import load_scorer,save_scorer
from .data import load_packet,arguments
from .rl import sample_modalities,path_logprob,group_loss
from training.qwen_scorer.grpo.train import build_reward

def native_domain(packet,record,prepared,patch):
    if record["target_raw_frame"]<=record["query_frame"] or not record.get("alignment_provenance"):
        raise ValueError("Causal target/provenance required")
    shape=prepared["latent_dict"]["targets"].shape
    frames=shape[2];per_frame=(shape[-2]//patch[1])*(shape[-1]//patch[2])
    result=[];modes=[];maps=[]
    for m,key,total,last in [(0,"video_candidate_map",frames*per_frame,per_frame)]:
        maps.append(validate_map(record[key],total,last,packet,m))
    tactile=prepared["action_dict"].get("tactile_global_targets")
    if tactile is not None and "tactile_candidate_map" not in record:
        raise ValueError("Tactile FM input requires an explicit tactile candidate map")
    if tactile is not None:
        _,s,c,f,h,w=tactile.shape
        if f!=frames:raise ValueError("Tactile temporal mismatch")
        # Sensor-major layout: each sensor's final frame is a target.
        p=(h//patch[1])*(w//patch[2])
        mapping=torch.as_tensor(record["tactile_candidate_map"],dtype=torch.long)
        if mapping.numel()!=s*f*p:raise ValueError("Tactile map length")
        parts=[validate_map(v.tolist(),f*p,p,packet,1) for v in mapping.reshape(s,f*p)]
        maps.append(torch.cat(parts))
    else:maps.append(torch.empty(0,dtype=torch.long))
    for m,mapping in enumerate(maps):
        valid=mapping>=0
        result.append(mapping[valid]);modes.append(torch.full((int(valid.sum()),),m,dtype=torch.long))
    return torch.cat(result),torch.cat(modes),maps

def validate_map(values,total,target_count,packet,modality):
    x=torch.as_tensor(values,dtype=torch.long)
    if x.shape!=(total,) or (x[:-target_count]<0).any() or (x[-target_count:]!=-1).any():
        raise ValueError("Map must cover all history and exclude exactly final target")
    v=x[x>=0]
    if (v>=len(packet["features"])).any():raise ValueError("Unknown feature row")
    if (packet["modalities"][v]!=modality).any():raise ValueError("Map modality mismatch")
    return x

def native_scores(paths,maps):
    sizes=[int((m>=0).sum()) for m in maps];offset=0;values=[]
    selected=torch.cat(paths).cpu()
    for m,size in zip(maps,sizes):
        v=torch.zeros(len(m))
        loc=torch.where(m>=0)[0]
        choice=selected[(selected>=offset)&(selected<offset+size)]-offset
        v[loc[choice]]=1.
        values.append(v);offset+=size
    return values

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(a.config).read_text())
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=False)
    device=cfg.get("device","cuda:0")
    from training.qwen_scorer.grpo.safety import require_idle_devices
    require_idle_devices(dict(qwen_device=device,n0_device=cfg["n0_device"]))
    model,info=load_scorer(cfg["checkpoint"],device)
    reference=deepcopy(model).eval().requires_grad_(False)
    reward=build_reward(cfg)
    rows=[json.loads(s) for s in Path(cfg["manifest"]).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError("Empty RL manifest")
    if cfg.get("group_size",4)!=4:raise ValueError("Initial group size is four")
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get("learning_rate",1e-5))
    rng=random.Random(cfg.get("seed",42));torch.manual_seed(cfg.get("seed",42))
    accumulation=int(cfg.get("gradient_accumulation_frames",4))
    if accumulation<1:raise ValueError("Invalid accumulation")
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    updates=0
    for epoch in range(cfg.get("epochs",1)):
        rng.shuffle(rows)
        for begin in range(0,len(rows),accumulation):
            group=rows[begin:begin+accumulation];optimizer.zero_grad(set_to_none=True)
            for r in group:
                packet=load_packet(r["feature_path"],info["encoder_revision"])
                args=arguments(packet,device)
                budgets=list(rng.choice(cfg["budget_choices"])) # video, action, tactile
                raw=torch.load(r["n0_batch"],map_location="cpu",weights_only=True)
                prepared=reward.prepare(raw,rng.randrange(2**31),rng.randrange(1000))
                index,modes,maps=native_domain(packet,r,prepared,reward.trainer.patch_size)
                index=index.to(device);modes=modes.to(device)
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                    model.eval();z=model(**args).float()[index]
                    ref=reference(**args).float()[index]
                    paths=[sample_modalities(z,modes,[budgets[0],budgets[2]]) for _ in range(4)]
                    old=torch.stack([path_logprob(z,modes,p) for p in paths])
                rewards=[]
                for path in paths:
                    v,t=native_scores(path,maps)
                    value,detail,selected=reward.evaluate(prepared,v,budgets,t if len(t) else None)
                    rewards.append(value)
                model.train()
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                    new=model(**args).float()[index]
                loss=group_loss(new,modes,paths,old,rewards)
                # Explicit logit anchoring, NOT mislabeled as subset-policy KL.
                anchor=0.
                for m in (0,1):
                    x=new[modes==m];y=ref[modes==m]
                    if len(x):anchor=anchor+((x-x.mean())-(y-y.mean())).square().mean()
                loss=loss+cfg.get("reference_logit_weight",.01)*anchor
                (loss/len(group)).backward()
                with (out/"metrics.jsonl").open("a") as f:
                    f.write(json.dumps(dict(id=r["id"],update=updates,rewards=rewards,
                       loss=float(loss.detach()),unique_sets=len({tuple(sorted(torch.cat(p).tolist())) for p in paths})))+"\n")
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError("Nonfinite RL gradient")
            optimizer.step();updates+=1
            save_scorer(model,out/f"scorer-{updates}.pt",encoder_revision=info["encoder_revision"])
            if updates>=cfg.get("max_updates",2**31):return
if __name__=="__main__":main()
