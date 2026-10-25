"""Online encoder timing using real RGB and pinned Qwen weights."""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .encoders import QwenEncoders
from .data import records
from .profile import summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--revision",required=True)
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--repeats",type=int,default=20)
    a=p.parse_args();torch.set_num_threads(8)
    r=records(a.manifest)[0];start=time.perf_counter()
    rgb=np.load(r["rgb_npy"],mmap_mode="r")
    images=[Image.fromarray(rgb[int(r["query_frame"]),c]) for c in range(3)]
    rgb_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();e=QwenEncoders(a.model,a.revision,a.device).eval()
    cuda=a.device.startswith("cuda")
    def sync():
        if cuda:torch.cuda.synchronize()
    sync();model_load_seconds=time.perf_counter()-start
    vision=[];text_cold=[];text_cached=[]
    with torch.inference_mode():
        for i in range(a.repeats+2):
            sync();t=time.perf_counter();f=e.images(images);sync()
            if i>=2:vision.append((time.perf_counter()-t)*1000)
            e._task_cache.clear()
            sync();t=time.perf_counter();v=e.task(r["task"]);sync()
            if i>=2:text_cold.append((time.perf_counter()-t)*1000)
            sync();t=time.perf_counter();v=e.task(r["task"]);sync()
            if i>=2:text_cached.append((time.perf_counter()-t)*1000)
    report=dict(model_revision=a.revision,device=torch.cuda.get_device_name() if cuda else "cpu",
        visual_parameters=sum(v.numel() for v in e.core.visual.parameters()),
        language_parameters=sum(v.numel() for v in e.core.language_model.parameters()),
        model_load_seconds=model_load_seconds,first_rgb_load_ms=rgb_ms,
        three_camera_frame_ms=summary(vision),task_uncached_ms=summary(text_cold),
        task_cached_ms=summary(text_cached),repeats=a.repeats,
        scope="Real 256x256 RGB, 3 cameras; includes preprocessing and image H2D. Excludes scorer, Sys1 and disk I/O except separately reported first RGB load.")
    with Path(a.output).open("x") as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)
if __name__=="__main__":main()
