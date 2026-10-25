"""Episode-level immutable features: encode each observed RGB frame once."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import records
from .encoders import QwenEncoders

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

def episode_packet(record,revision=None):
    """Read only the query prefix, never future frames, from the immutable bank."""
    from .data import targets,validate_packet
    directory=Path(record["episode_features"])
    meta=json.loads((directory/"COMPLETE.json").read_text())
    if revision is not None and meta["encoder_revision"]!=revision:
        raise ValueError("Stale episode feature cache")
    if meta["episode_id"]!=record["episode_id"] or meta["task"]!=record["task"]:
        raise ValueError("Episode/task mismatch")
    n=int(record["query_frame"])+1
    if n<1 or n>meta["frames"] or float(record["fps"])!=meta["fps"]:
        raise ValueError("Invalid query prefix")
    mm=np.memmap(directory/"features.bf16",dtype=np.uint16,mode="r",
                 shape=(meta["frames"]*192,meta["vision_dim"]))
    # uint16 storage preserves BF16 bits exactly (no FP16 conversion).
    features=torch.from_numpy(np.array(mm[:n*192],copy=True)).view(torch.bfloat16)
    del mm
    text=torch.load(directory/"text.pt",map_location="cpu",weights_only=True)
    frames=torch.arange(n).repeat_interleave(192)
    sensors=torch.arange(3).repeat_interleave(64).repeat(n)
    xy=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing="ij"),-1).reshape(-1,2)/7.
    p=dict(features=features,text=text,times=frames.float()/meta["fps"],
           xy=xy.repeat(n*3,1),modalities=torch.zeros(n*192,dtype=torch.long),
           sensors=sensors,query_time=(n-1)/meta["fps"],
           ids=[f'{record["episode_id"]}:f{f}:c{c}:p{k}' for f in range(n) for c in range(3) for k in range(64)],
           encoder_revision=meta["encoder_revision"])
    if "target" in record:p["targets"]=targets(record)
    validate_packet(p);return p

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifests",nargs="+",required=True)
    p.add_argument("--model",required=True);p.add_argument("--revision",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--image-batch",type=int,default=8);p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.image_batch<1:raise ValueError("Positive image batch required")
    out=Path(a.output)
    spec=dict(model=str(Path(a.model).resolve()),revision=a.revision,
              manifests={str(Path(v).resolve()):digest(v) for v in a.manifests},
              format="rgb-frame-camera-patch-bf16-linear-projection-v1")
    if out.exists():
        if not a.resume or json.loads((out/"config.json").read_text())!=spec:
            raise ValueError("Output exists or incompatible resume")
    else:
        out.mkdir(parents=True);write_json(out/"config.json",spec)
    manifests=[records(v) for v in a.manifests];episodes={}
    for rows in manifests:
        for r in rows:
            group=episodes.setdefault(r["episode_id"],[]);group.append(r)
    encoder=None
    started=time.perf_counter();processed_tokens=0
    for eid,rows in sorted(episodes.items()):
        r=rows[0];directory=out/"episodes"/hashlib.sha256(eid.encode()).hexdigest()
        frames=max(int(x["query_frame"])+1 for x in rows)
        if len({(x["rgb_npy"],x["task"],float(x["fps"])) for x in rows})!=1:
            raise ValueError("Inconsistent episode metadata")
        if (directory/"COMPLETE.json").exists():
            m=json.loads((directory/"COMPLETE.json").read_text())
            if m["encoder_revision"]!=a.revision or m["frames"]!=frames or m["task"]!=r["task"]:
                raise ValueError("Incompatible completed episode")
            if (directory/"features.bf16").stat().st_size!=frames*192*m["vision_dim"]*2:
                raise ValueError("Truncated episode feature file")
            continue
        if encoder is None:
            encoder=QwenEncoders(a.model,a.revision,a.device,image_batch=a.image_batch).eval()
        episode_start=time.perf_counter()
        if a.device.startswith("cuda"):torch.cuda.reset_peak_memory_stats()
        directory.mkdir(parents=True,exist_ok=True)
        rgb=np.load(r["rgb_npy"],mmap_mode="r")
        if rgb.dtype!=np.uint8 or rgb.shape[1:]!=(3,256,256,3) or len(rgb)<frames:
            raise ValueError("Invalid RGB array")
        mm=np.memmap(directory/"features.bf16",mode="w+",dtype=np.uint16,
                     shape=(frames*192,encoder.vision_dim))
        with torch.inference_mode():
            for begin in range(0,frames*3,a.image_batch):
                end=min(begin+a.image_batch,frames*3)
                images=[Image.fromarray(rgb[j//3,j%3]) for j in range(begin,end)]
                encoded=encoder.images(images).to(dtype=torch.bfloat16).cpu().contiguous()
                if not torch.isfinite(encoded).all():raise FloatingPointError("Nonfinite encoder output")
                mm[begin*64:end*64]=encoded.view(torch.uint16).numpy()
            task=encoder.task(r["task"]).detach().cpu().clone()
        mm.flush();del mm,rgb
        torch.save(task,directory/"text.pt")
        write_json(directory/"COMPLETE.json",dict(episode_id=eid,task=r["task"],
            frames=frames,fps=float(r["fps"]),vision_dim=encoder.vision_dim,
            text_dim=encoder.text_dim,encoder_revision=a.revision))
        print(json.dumps(dict(event="episode_complete",episode_id=eid,frames=frames,
                              tokens=frames*192,seconds=time.perf_counter()-episode_start,
                              peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith("cuda") else None)),flush=True)
        processed_tokens+=frames*192
    outputs=[]
    for i,(source,rows) in enumerate(zip(a.manifests,manifests)):
        target=out/(Path(source).stem+".features.jsonl")
        if target in outputs:raise ValueError("Manifest filename collision")
        outputs.append(target)
        result=[]
        for r in rows:
            d=out/"episodes"/hashlib.sha256(r["episode_id"].encode()).hexdigest()
            result.append(dict(r,episode_features=str(d.resolve()),encoder_revision=a.revision))
        tmp=target.with_suffix(".tmp");tmp.write_text("".join(json.dumps(r)+"\n" for r in result))
        os.replace(tmp,target)
    write_json(out/"COMPLETE.json",dict(episodes=len(episodes),manifests=[str(v) for v in outputs],
                                      encoder_revision=a.revision,wall_seconds=time.perf_counter()-started,
                                      processed_tokens_this_run=processed_tokens,
                                      feature_bytes=sum(v.stat().st_size for v in (out/"episodes").glob("*/features.bf16"))))
if __name__=="__main__":main()
