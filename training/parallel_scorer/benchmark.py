"""Synthetic cached-feature scorer benchmark, NOT end-to-end robot latency."""
import argparse,json,time,platform,statistics,subprocess
from contextlib import nullcontext
import torch
from .model import ScorerConfig,ParallelScorer,select_indices

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--preset",choices=["small","medium"],default="small")
    p.add_argument("--tokens",type=int,nargs="+",default=[1024,4096,10000,20000,40000])
    p.add_argument("--device",default="cuda:0");p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--warmup",type=int,default=2);p.add_argument("--query-chunk",type=int,default=2048)
    p.add_argument("--vision-dim",type=int,default=1024);p.add_argument("--text-dim",type=int,default=1024)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    cuda=a.device.startswith("cuda")
    if cuda:
        gpu=a.device.split(":")[-1]
        import os
        visible=os.environ.get("CUDA_VISIBLE_DEVICES")
        physical=visible.split(",")[int(gpu)] if visible else gpu
        active=subprocess.check_output(["nvidia-smi","-i",physical,
             "--query-compute-apps=pid","--format=csv,noheader"],text=True)
        if any(s.strip().isdigit() for s in active.splitlines()):
            raise RuntimeError("Selected GPU has existing compute processes; do not benchmark a training GPU")
    torch.manual_seed(7)
    torch.set_num_threads(4)
    c=ScorerConfig.preset(a.preset,vision_dim=a.vision_dim,text_dim=a.text_dim,
                         query_chunk=a.query_chunk,flash_only=cuda)
    model=ParallelScorer(c).eval().to(a.device)
    if cuda:model=model.to(torch.bfloat16)
    dtype=next(model.parameters()).dtype
    rows=[]
    def sync():
        if cuda:torch.cuda.synchronize(a.device)
    with torch.inference_mode():
        for n in a.tokens:
            f=torch.randn(n,c.vision_dim,device=a.device,dtype=dtype)
            text=torch.randn(32,c.text_dim,device=a.device,dtype=dtype)
            times=torch.arange(n,device=a.device).float()//192
            xy=torch.rand(n,2,device=a.device)
            mods=torch.arange(n,device=a.device)%2;sensors=torch.arange(n,device=a.device)%3
            def run():
                scores=model(f,text,times,xy,mods,sensors,times.max())
                return select_indices(scores,mods,[min(n//4,2048)]*2)
            for _ in range(a.warmup):run()
            sync()
            if cuda:torch.cuda.reset_peak_memory_stats(a.device)
            elapsed=[]
            for _ in range(a.repeats):
                sync();start=time.perf_counter();run();sync()
                elapsed.append((time.perf_counter()-start)*1000)
            ordered=sorted(elapsed)
            row=dict(tokens=n,median_ms=statistics.median(elapsed),
                p95_ms=ordered[min(len(ordered)-1,int(.95*len(ordered)))],
                min_ms=min(elapsed),samples_ms=elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if cuda else None)
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(preset=a.preset,counts=model.counts(),device=torch.cuda.get_device_name(a.device) if cuda else platform.processor(),
                torch=torch.__version__,dtype=str(dtype),query_chunk=a.query_chunk,text_tokens=32,
                scope="random cached features + full scorer + top-k, excludes RGB/text encoders, host transfer, Sys1 KV fetch and robot",
                rows=rows)
    from pathlib import Path
    Path(a.output).write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
