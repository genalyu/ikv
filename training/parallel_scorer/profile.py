"""Real-feature scoring latency, split into I/O, transfer, scoring and selection."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from .data import records,load_record,arguments
from .model import load_scorer,select_indices

def summary(values):
    ordered=sorted(values)
    return dict(p50=statistics.median(values),p95=ordered[min(len(ordered)-1,int(.95*len(ordered)))])
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True);p.add_argument("--manifest",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--output",required=True)
    p.add_argument("--repeats",type=int,default=20);p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--visual-budget",type=int,default=2048);p.add_argument("--tactile-budget",type=int,default=1024)
    a=p.parse_args()
    torch.set_num_threads(8)
    rows=records(a.manifest);rows.sort(key=lambda r:int(r.get("query_frame",0)))
    chosen=[]
    for quantile in [0.,.5,.9,.99,1.]:
        r=rows[int((len(rows)-1)*quantile)]
        if r["id"] not in {v["id"] for v in chosen}:chosen.append(r)
    model,info=load_scorer(a.checkpoint,a.device);model.eval()
    cuda=a.device.startswith("cuda")
    if cuda:model.to(torch.bfloat16)
    def sync():
        if cuda:torch.cuda.synchronize(a.device)
    results=[]
    with torch.inference_mode():
        for r in chosen:
            t=time.perf_counter();packet=load_record(r,info["encoder_revision"])
            load_ms=(time.perf_counter()-t)*1000
            transfers=[];scoring=[];selection=[];returns=[];total=[]
            if cuda:torch.cuda.reset_peak_memory_stats()
            for i in range(a.warmup+a.repeats):
                sync();start=time.perf_counter();args=arguments(packet,a.device)
                sync();t1=time.perf_counter()
                z=model(**args)
                sync();t2=time.perf_counter()
                idx=select_indices(z,args["modalities"],[a.visual_budget,a.tactile_budget])
                sync();t3=time.perf_counter();ids=idx.cpu()
                sync();end=time.perf_counter()
                if i>=a.warmup:
                    transfers.append((t1-start)*1000);scoring.append((t2-t1)*1000)
                    selection.append((t3-t2)*1000);returns.append((end-t3)*1000)
                    total.append((end-start)*1000)
                del args,z,idx,ids
            results.append(dict(id=r["id"],tokens=len(packet["features"]),first_packet_load_ms=load_ms,
                transfer_ms=summary(transfers),scorer_ms=summary(scoring),selection_ms=summary(selection),
                selected_indices_return_ms=summary(returns),resident_cpu_packet_to_selection_ms=summary(total),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None))
            print(json.dumps(results[-1]),flush=True)
    with Path(a.output).open("x") as f:
        json.dump(dict(device=torch.cuda.get_device_name(a.device) if cuda else "CPU",
            counts=model.counts(),repeats=a.repeats,rows=results,
            scope="Real cached features; excludes RGB/text encoding and Sys1 KV fetch/control. First packet I/O separate; CPU packet reused across timed repeats."),f,indent=2)
if __name__=="__main__":main()
