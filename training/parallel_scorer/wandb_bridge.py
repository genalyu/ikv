"""CPU-side live W&B uploader for parallel scorer experiments.
Reads durable JSON logs; does not import models or modify training processes.
"""
import argparse,fcntl,hashlib,json,os,signal,socket,time
from pathlib import Path

SPECS=[("0.8B","small"),("0.8B","medium"),("2B","small"),("2B","medium")]
def rows(path):
    if not Path(path).exists():return []
    result=[]
    for line in Path(path).read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(r,dict):result.append(r)
    return result

def counts(size,preset):
    d,l=(512,4) if preset=="small" else (768,6)
    feature=1024 if size=="0.8B" else 2048
    interaction=l*(16*d*d+19*d)
    return dict(interaction=interaction,total=interaction+2*feature*d+3*d*d//2+47*d+1)

def update_metrics(row):
    if row.get("event")=="update":
        mapping=dict(loss="train/score_loss",grad_norm="train/grad_norm",tokens="train/tokens",
          seconds="time/update_seconds",data_seconds="time/data_seconds",
          training_seconds="cost/training_seconds",training_gpu_hours="cost/training_gpu_hours",
          peak_gpu_gib="memory/peak_allocated_gib")
        result={"optimizer_step":int(row["update"]),"epoch":int(row["epoch"])}
        result.update({dst:row[src] for src,dst in mapping.items() if row.get(src) is not None})
        if row.get("training_seconds") is not None:result["training_elapsed_hours"]=row["training_seconds"]/3600
        return result
    if row.get("event")=="validation":
        return dict(optimizer_step=int(row["update"]),epoch=int(row["epoch"]),
            **{"validation/teacher_mae":row["mae"],"time/validation_seconds":row["seconds"],
               "validation/queries":row["queries"]})
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True);p.add_argument("--entity",required=True)
    p.add_argument("--project",required=True);p.add_argument("--poll",type=int,default=5)
    a=p.parse_args();root=Path(a.root)
    out=root/"wandb-bridge";out.mkdir(exist_ok=True)
    lock=(out/"bridge.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["WANDB_BASE_URL"]="https://api.wandb.ai"
    os.environ["WANDB_DIR"]=str(out)
    os.environ["WANDB_CACHE_DIR"]=str(out/"cache")
    os.environ["WANDB_CONFIG_DIR"]=str(out/"config")
    os.environ["WANDB_DISABLE_GIT"]="true"
    import wandb
    statefile=out/"state.json"
    state=json.loads(statefile.read_text()) if statefile.exists() else {"runs":{}}
    source=json.loads((root/"SOURCE_STATE.json").read_text())
    plan=json.loads((root/"EXPERIMENT_PLAN.json").read_text())
    group="parallel-scorer-"+root.name.rsplit("-",1)[-1]
    job=json.loads((root/"CACHE_JOBS.json").read_text())
    origin=job["start"]
    runs={}
    def init(key,name,config,job_type):
        runid="ikv"+hashlib.sha256((str(root.resolve())+key).encode()).hexdigest()[:16]
        run=wandb.init(entity=a.entity,project=a.project,id=runid,resume="allow",
            name=name,group=group,job_type=job_type,config=config,
            tags=["parallel-scorer",group],reinit="create_new",
            settings=wandb.Settings(mode="online",base_url="https://api.wandb.ai",x_disable_stats=True,init_timeout=60))
        runs[key]=run
        state["runs"].setdefault(key,dict(id=runid,metrics_index=0,finished_files=[],last_gpu_time=0,cache_index=0))
        state["runs"][key]["url"]=run.url
        run.define_metric("optimizer_step",hidden=True)
        run.define_metric("training_elapsed_hours",hidden=True)
        run.define_metric("inference/history_tokens",hidden=True)
        run.define_metric("telemetry/elapsed_hours",hidden=True)
        run.define_metric("cache/elapsed_hours",hidden=True)
        for pattern in ["train/*","validation/*","time/*","memory/*","cost/*"]:
            run.define_metric(pattern,step_metric="optimizer_step")
        run.define_metric("inference/*",step_metric="inference/history_tokens")
        run.define_metric("hardware/*",step_metric="telemetry/elapsed_hours")
        run.define_metric("cache/*",step_metric="cache/elapsed_hours")
        return run
    for size,preset in SPECS:
        key="qwen"+size+"-"+preset
        c=counts(size,preset)
        run=init(key,f"Qwen{size} + {'17M' if preset=='small' else '57M'}",
            dict(encoder=size,interaction_preset=preset,interaction_parameters=c["interaction"],
                scorer_parameters=c["total"],encoder_frozen=True,architecture="bidirectional_parallel_scorer",
                epochs=3,seed=42,gradient_accumulation=4,learning_rate=1e-4,
                train_queries=6809,validation_queries=830,test_queries=886,test_used=False,
                source_commit=source["commit"],hardware="A100 80GB",
                features="full history; episode cache; bf16; linear patch projection",
                teacher="existing Qwen-Max RGB labels",metric_semantics="score regression + pairwise ranking; unlike old 4B JSON CE",
                run_root=str(root),shared_preencoding=True), "scorer-sft")
        run.summary["phase"]="waiting_for_preencoding" if size=="0.8B" else "queued"
        run.summary["training_started"]=False
    infra=init("shared-costs","Shared preencoding + GPU utilization",
        dict(source_commit=source["commit"],hardware="2 x A100 80GB",model_run_group=group,
             accounting="Shared preprocessing is recorded once here, not duplicated as per-model training cost"),
        "shared-preencoding")
    stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    def save():
        state["updated"]=time.time()
        state["entity"]=a.entity;state["project"]=a.project;state["group"]=group
        temp=statefile.with_suffix(".tmp");temp.write_text(json.dumps(state,indent=2));os.replace(temp,statefile)
        status=dict(entity=a.entity,project=a.project,group=group,project_url=f"https://wandb.ai/{a.entity}/{a.project}",
          runs={k:dict(url=v["url"],uploaded_metric_rows=v["metrics_index"]) for k,v in state["runs"].items()},
          pid=os.getpid(),time=time.time(),host=socket.gethostname(),poll_seconds=a.poll)
        (root/"WANDB_STATUS.json").write_text(json.dumps(status,indent=2))
    save();print(json.dumps(dict(event="online",project=a.project,group=group)),flush=True)
    try:
        while not stop[0]:
            pipeline=json.loads((root/"PIPELINE_STATUS.json").read_text())
            for size,preset in SPECS:
                key="qwen"+size+"-"+preset;run=runs[key];s=state["runs"][key];folder=root/"runs"/key
                history=rows(folder/"metrics.jsonl")
                if s["metrics_index"]>len(history):raise ValueError("Metric history was truncated")
                for row in history[s["metrics_index"]:]:
                    values=update_metrics(row)
                    if values:run.log(values)
                    if row.get("event")=="start":run.config.update({"scorer_parameters":row["counts"]["total"]},allow_val_change=True)
                    s["metrics_index"]+=1
                phase=pipeline.get(key+"-train",{}).get("status")
                if history:
                    run.summary["training_started"]=True
                    run.summary["latest_optimizer_step"]=max((v.get("update",0) for v in history),default=0)
                run.summary["phase"]=phase or ("waiting_for_preencoding" if size=="0.8B" else "queued")
                for name in ["COMPLETE.json","validation-evaluation.json","inference-latency.json"]:
                    path=folder/name
                    if not path.exists() or name in s["finished_files"]:continue
                    data=json.loads(path.read_text())
                    if name=="COMPLETE.json":
                        run.summary["phase"]="training_complete"
                        for k in ["best_validation_mae","updates","training_seconds","validation_seconds"]:
                            if k in data:run.summary["final/"+k]=data[k]
                    elif name=="validation-evaluation.json":
                        for modality,metrics in data["summary"].items():
                            for k,v in metrics.items():run.summary[f"quality/{modality}/{k}"]=v
                    else:
                        table=[]
                        for r in data["rows"]:
                            entry={"inference/history_tokens":r["tokens"]}
                            for category in ["transfer_ms","scorer_ms","selection_ms","selected_indices_return_ms","resident_cpu_packet_to_selection_ms"]:
                                for q,v in r[category].items():entry[f"inference/{category}_{q}"]=v
                            run.log(entry)
                            table.append([r["tokens"],r["first_packet_load_ms"],
                              r["resident_cpu_packet_to_selection_ms"]["p50"],r["resident_cpu_packet_to_selection_ms"]["p95"],
                              r["scorer_ms"]["p50"],r.get("peak_allocated_gib")])
                        run.log({"latency_table":wandb.Table(columns=["history_tokens","first_packet_IO_ms","cpu_packet_to_selection_p50_ms","cpu_packet_to_selection_p95_ms","scorer_p50_ms","peak_VRAM_GiB"],data=table)})
                    s["finished_files"].append(name)
                enc=root/"results"/("encoder-"+size+".json")
                if enc.exists() and "encoder_profile" not in s["finished_files"]:
                    data=json.loads(enc.read_text())
                    for category in ["three_camera_frame_ms","task_uncached_ms","task_cached_ms"]:
                        for q,v in data[category].items():run.summary[f"encoder/{category}_{q}"]=v
                    for k in ["visual_parameters","language_parameters","model_load_seconds"]:
                        run.summary["encoder/"+k]=data[k]
                    s["finished_files"].append("encoder_profile")
                stage=pipeline.get(key+"-train",{})
                if "wall_seconds" in stage:run.summary["cost/full_training_stage_wall_hours"]=stage["wall_seconds"]/3600
            ins=state["runs"]["shared-costs"]
            hardware=rows(root/"hardware_metrics.jsonl")
            for row in hardware:
                if row["time"]<=ins["last_gpu_time"]:continue
                vals={"telemetry/elapsed_hours":max(0,row["time"]-origin)/3600}
                for g in row["gpus"]:
                    for key,value in g.items():
                        if key!="index":vals[f'hardware/gpu{g["index"]}/{key}']=value
                infra.log(vals);ins["last_gpu_time"]=row["time"]
            cache=[]
            for i in range(2):
                cache.extend(dict(v,shard=i,encoder="0.8B") for v in rows(root/f"logs/cache-0.8B-linear-shard{i}.log") if v.get("event")=="episode_complete")
            cache.extend(dict(v,shard=0,encoder="2B") for v in rows(root/"logs/qwen2B-cache.log") if v.get("event")=="episode_complete")
            # Log progress snapshots rather than pretending per-episode ordering is wall-clock ordering.
            total=len(cache)
            if total!=ins["cache_index"]:
                values={"cache/elapsed_hours":max(0,time.time()-origin)/3600}
                for size in ["0.8B","2B"]:
                    subset=[v for v in cache if v["encoder"]==size]
                    values[f"cache/{size}/episodes"]=len(subset)
                    values[f"cache/{size}/tokens"]=sum(v["tokens"] for v in subset)
                    values[f"cache/{size}/encoding_gpu_hours"]=sum(v["seconds"] for v in subset)/3600
                infra.log(values);ins["cache_index"]=total
            for size,paths in [("0.8B",[root/f"features-0.8B-linear/shard{i}/COMPLETE.json" for i in range(2)]),
                               ("2B",[root/"features-2B-linear/COMPLETE.json"])]:
                if all(p.exists() for p in paths):
                    values=[json.loads(p.read_text()) for p in paths]
                    infra.summary[f"cache/{size}/bytes"]=sum(v["feature_bytes"] for v in values)
                    infra.summary[f"cache/{size}/encoding_phase_gpu_hours"]=sum(v["wall_seconds"] for v in values)/3600
                    infra.summary[f"cache/{size}/encoding_phase_wall_hours"]=max(v["wall_seconds"] for v in values)/3600
            infra.summary["pipeline_phase"]=pipeline["pipeline"]["status"]
            save()
            if pipeline["pipeline"]["status"]=="complete":break
            time.sleep(a.poll)
    finally:
        save()
        for run in runs.values():run.finish()
if __name__=="__main__":main()
