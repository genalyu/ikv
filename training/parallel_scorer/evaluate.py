"""Offline teacher agreement; not a replacement for closed-loop task success."""
import argparse,json,time
from pathlib import Path
import torch
from .data import records,load_packet,arguments
from .model import load_scorer

def metrics(pred,target,k):
    pred=pred.float();target=target.float()
    k=min(k,len(pred))
    if k<1:raise ValueError("Positive k and nonempty candidates required")
    actual=torch.argsort(pred,descending=True,stable=True)[:k]
    ideal=torch.argsort(target,descending=True,stable=True)[:k]
    discount=1/torch.log2(torch.arange(k,device=pred.device).float()+2)
    # Linear gain avoids treating a teacher's arbitrary 0..100 scale exponentially.
    dcg=(target[actual]*discount).sum();idcg=(target[ideal]*discount).sum()
    return dict(mae=float((pred-target).abs().mean()),
        ndcg_linear=None if idcg<=0 else float(dcg/idcg),
        selected_teacher_mean=float(target[actual].mean()),
        ideal_teacher_mean=float(target[ideal].mean()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True);p.add_argument("--manifest",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--output",required=True)
    p.add_argument("--k",type=int,default=1024)
    a=p.parse_args();model,info=load_scorer(a.checkpoint,a.device);model.eval()
    rows=[]
    with torch.inference_mode():
        for r in records(a.manifest):
            packet=load_packet(r["feature_path"],info["encoder_revision"])
            args=arguments(packet,a.device)
            if a.device.startswith("cuda"):torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=a.device.startswith("cuda")):
                z=model(**args).float().sigmoid()*100
            if a.device.startswith("cuda"):torch.cuda.synchronize()
            row=dict(id=r["id"],episode_id=r["episode_id"],
                     scorer_ms=(time.perf_counter()-start)*1000,modalities={})
            mask=packet.get("supervised",torch.ones(len(z),dtype=torch.bool)).to(a.device).bool()
            y=packet["targets"].to(a.device);mods=packet["modalities"].to(a.device)
            for m,name in [(0,"visual"),(1,"tactile")]:
                valid=mask & (mods==m)
                if valid.any():row["modalities"][name]=metrics(z[valid],y[valid],a.k)
            rows.append(row)
    summary={}
    for name in ("visual","tactile"):
        values=[r["modalities"][name] for r in rows if name in r["modalities"]]
        if values:
            summary[name]={key:sum(v[key] for v in values if v[key] is not None)/
                sum(v[key] is not None for v in values)
                for key in values[0] if any(v[key] is not None for v in values)}
    with Path(a.output).open("x") as f:
        json.dump(dict(scope="Teacher agreement only; macro-average per query",
                       summary=summary,rows=rows),f,indent=2)
if __name__=="__main__":main()
