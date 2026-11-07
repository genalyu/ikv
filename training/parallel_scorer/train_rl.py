"""Group-relative native-KV selection with the existing frozen N0 FM evaluator.
Manifest: feature_path plus n0_batch, target_raw_frame, query_frame,
alignment_provenance, video_candidate_map and optional tactile_candidate_map.
Each map covers ALL clean native tokens (including target); values are feature
row indices, or -1 ONLY for the final target tokens. Mapping is explicit.
"""
import argparse,json,random,os,time,hashlib
import torch.distributed as dist
from pathlib import Path
from copy import deepcopy
import torch
from .model import load_scorer,save_scorer
from .data import load_record,arguments
from .rl import sample_modalities,path_logprob,group_loss,SensitivityWeights
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
        if mapping.shape[0]!=s*f*p:raise ValueError("Tactile map length")
        parts=[validate_map(v.tolist(),f*p,p,packet,1) for v in mapping.reshape(s,f*p,-1)]
        maps.append(torch.cat(parts))
    else:maps.append(torch.empty((0,4),dtype=torch.long))
    for m,mapping in enumerate(maps):
        valid=mapping>=0
        if mapping.ndim == 2: valid = valid.all(-1)
        result.append(mapping[valid]);modes.append(torch.full((int(valid.sum()),),m,dtype=torch.long))
    return torch.cat(result),torch.cat(modes),maps

def validate_map(values,total,target_count,packet,modality):
    x=torch.as_tensor(values,dtype=torch.long)
    if x.ndim == 1: x=x[:,None]
    if x.ndim == 2 and x.shape[1] == 1: x=x.expand(-1,4)
    if x.shape!=(total,4) or (x[:-target_count]<0).any() or (x[-target_count:]!=-1).any():
        raise ValueError("Map must cover all history and exclude exactly final target")
    v=x[x>=0]
    if (v>=len(packet["features"])).any():raise ValueError("Unknown feature row")
    if (packet["modalities"][v]!=modality).any():raise ValueError("Map modality mismatch")
    return x

def native_scores(paths,maps):
    sizes=[int((m>=0).all(-1).sum()) for m in maps];offset=0;values=[]
    selected=torch.cat(paths).cpu()
    for m,size in zip(maps,sizes):
        v=torch.zeros(len(m))
        loc=torch.where((m>=0).all(-1))[0]
        choice=selected[(selected>=offset)&(selected<offset+size)]-offset
        v[loc[choice]]=1.
        values.append(v);offset+=size
    return values

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(a.config).read_text())
    out=Path(cfg["output"])
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    device=f"cuda:{os.environ.get('LOCAL_RANK',0)}" if world>1 else cfg.get("device","cuda:0")
    if world>1:cfg["n0_device"]=device
    from training.qwen_scorer.grpo.safety import require_idle_devices
    require_idle_devices(dict(qwen_device=device,n0_device=cfg["n0_device"]))
    if world>1:
        torch.cuda.set_device(device);dist.init_process_group("nccl")
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    model,info=load_scorer(cfg["checkpoint"],device)
    reference=deepcopy(model).eval().requires_grad_(False)
    reward=build_reward(cfg)
    rows=[json.loads(s) for s in Path(cfg["manifest"]).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError("Empty RL manifest")
    if cfg.get("held_out_manifest"):
        from .data import check_disjoint
        held_out=[json.loads(s) for s in Path(cfg["held_out_manifest"]).read_text().splitlines() if s.strip()]
        check_disjoint(rows,held_out)
    if cfg.get("group_size",4)!=4:raise ValueError("Initial group size is four")
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get("learning_rate",1e-5))
    rng=random.Random(cfg.get("seed",42)+rank);shuffle_rng=random.Random(cfg.get("seed",42));torch.manual_seed(cfg.get("seed",42)+rank)
    accumulation=int(cfg.get("gradient_accumulation_frames",4))
    if accumulation<1:raise ValueError("Invalid accumulation")
    wb=None
    if rank==0:
        (out/"config.json").write_text(json.dumps(cfg,indent=2))
        if cfg.get("wandb",{}).get("enabled",False):
            import wandb
            options=cfg["wandb"]
            wb=wandb.init(project=options["project"],entity=options.get("entity"),
                name=options.get("name"),config=dict(cfg,world_size=world),dir=str(out),mode="online")
            (out/"WANDB.json").write_text(json.dumps(dict(url=wb.url,id=wb.id)))
    sensitivity=SensitivityWeights(
        decay=cfg.get("sensitivity_ema_decay",.95),
        minimum=cfg.get("sensitivity_weight_min",.25),
        maximum=cfg.get("sensitivity_weight_max",4.))
    updates=0
    for epoch in range(cfg.get("epochs",1)):
        shuffle_rng.shuffle(rows)
        usable=(len(rows)//(world*accumulation))*(world*accumulation) if world>1 else len(rows)
        if not usable:raise ValueError("Not enough queries for one distributed update")
        for begin in range(0,usable,accumulation*world):
            start=begin+rank*accumulation
            group=rows[start:start+accumulation];optimizer.zero_grad(set_to_none=True)
            update_start=time.monotonic()
            pending=[];audits=[]
            for r in group:
                packet=load_record(r,info["encoder_revision"])
                args=arguments(packet,device)
                budgets=list(rng.choice(cfg["budget_choices"])) # video, action, tactile
                raw=torch.load(r["n0_batch"],map_location="cpu",weights_only=True)
                noise_seed,time_index=rng.randrange(2**31),rng.randrange(1000)
                prepared=reward.prepare(raw,noise_seed,time_index)
                index,modes,maps=native_domain(packet,r,prepared,reward.trainer.patch_size)
                index=index.to(device);modes=modes.to(device)
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                    model.eval();z=model(**args).float()[index].mean(-1)
                    ref=reference(**args).float()[index].mean(-1)
                    paths=[sample_modalities(z,modes,[budgets[0],budgets[2]]) for _ in range(4)]
                    old=torch.stack([path_logprob(z,modes,p) for p in paths])
                rewards=[];fm=[]
                for path in paths:
                    v,t=native_scores(path,maps)
                    value,detail,selected=reward.evaluate(prepared,v,budgets,t if len(t) else None)
                    rewards.append(value);fm.append(detail)
                pending.append((r,packet,index,modes,paths,old,ref,rewards))
                audits.append(dict(budgets=budgets,noise_seed=noise_seed,time_index=time_index,fm=fm,
                    unique_sets=len({tuple(sorted(torch.cat(p).tolist())) for p in paths})))
            all_rewards=[item[-1] for item in pending]
            if world>1:
                gathered=[None]*world;dist.all_gather_object(gathered,all_rewards)
                all_rewards=[r for part in gathered for r in part]
            fractions,spreads,raw_weights=sensitivity.weights(all_rewards)
            offset=rank*accumulation
            fractions=fractions[offset:offset+len(group)]*world
            spreads=spreads[offset:offset+len(group)];raw_weights=raw_weights[offset:offset+len(group)]
            for query_index,(r,packet,index,modes,paths,old,ref,rewards) in enumerate(pending):
                args=arguments(packet,device)
                model.train()
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                    new=model(**args).float()[index].mean(-1)
                loss=group_loss(new,modes,paths,old,rewards)
                # Explicit logit anchoring, NOT mislabeled as subset-policy KL.
                anchor=0.
                for m in (0,1):
                    x=new[modes==m];y=ref[modes==m]
                    if len(x):anchor=anchor+((x-x.mean())-(y-y.mean())).square().mean()
                policy_loss=loss
                loss=fractions[query_index].to(device)*policy_loss+cfg.get("reference_logit_weight",.01)*anchor/len(group)
                loss.backward()
                with (out/("metrics.jsonl" if rank==0 else f"metrics-rank{rank}.jsonl")).open("a") as f:
                    f.write(json.dumps(dict(id=r["id"],update=updates,rewards=rewards,
                       loss=float(loss.detach()),sensitivity=float(spreads[query_index]),
                       query_weight=float(raw_weights[query_index]),query_fraction=float(fractions[query_index])/world,
                       sensitivity_ema=sensitivity.reference,**audits[query_index]))+"\n")
            if world>1:
                for parameter in model.parameters():
                    if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad);parameter.grad.div_(world)
            all_audits=audits
            if world>1:
                collected=[None]*world;dist.all_gather_object(collected,audits)
                all_audits=[entry for part in collected for entry in part]
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError("Nonfinite RL gradient")
            optimizer.step();updates+=1
            if world>1 and updates==1:
                digest=hashlib.sha256()
                for parameter in model.parameters():digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
                signatures=[None]*world;dist.all_gather_object(signatures,digest.hexdigest())
                if len(set(signatures))!=1:raise RuntimeError("Scorer replicas diverged")
                if rank==0:(out/"SYNC_VERIFIED.json").write_text(json.dumps(dict(update=updates,world_size=world,sha256=signatures[0])))
            if rank==0:
                summary=dict(update=updates,reward_mean=float(torch.tensor(all_rewards).mean()),
                    sensitivity_mean=float(torch.tensor(all_rewards).std(-1,unbiased=False).mean()),
                    grad_norm=float(norm),seconds=time.monotonic()-update_start,queries=len(all_rewards),
                    unique_sets_mean=sum(x['unique_sets'] for x in all_audits)/len(all_audits))
                for modality in ('video','action','tactile'):
                    summary['fm/'+modality]=sum(d[modality] for a in all_audits for d in a['fm'])/(4*len(all_audits))
                with (out/"updates.jsonl").open("a") as f:f.write(json.dumps(summary)+"\n")
                print(json.dumps(summary),flush=True)
                if wb:wb.log(summary,step=updates)
                if updates%cfg.get("save_every",10)==0 or updates==cfg.get("max_updates"):
                    save_scorer(model,out/f"scorer-{updates}.pt",encoder_revision=info["encoder_revision"],
                        sensor_to_id=info.get("sensor_to_id"),sensitivity_reference=sensitivity.reference)
                    torch.save(dict(update=updates,optimizer=optimizer.state_dict(),
                        sensitivity_reference=sensitivity.reference),out/f"optimizer-{updates}.pt")
            if updates>=cfg.get("max_updates",2**31):
                if wb:wb.finish()
                if world>1:dist.destroy_process_group()
                return
    if rank==0:
        save_scorer(model,out/f"scorer-{updates}.pt",encoder_revision=info["encoder_revision"],
            sensor_to_id=info.get("sensor_to_id"),sensitivity_reference=sensitivity.reference)
        if wb:wb.finish()
    if world>1:dist.destroy_process_group()
if __name__=="__main__":main()
