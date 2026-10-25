"""Encode immutable feature packets once per query (offline convenience).
For streaming deployments use FeatureBank, encoding only newly arrived frames.
"""
import argparse,json,hashlib
from pathlib import Path
import torch
from .encoders import QwenEncoders
from .data import records
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True);p.add_argument("--model",required=True)
    p.add_argument("--revision",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--encoder-state")
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    encoder=QwenEncoders(a.model,a.revision,a.device).eval()
    if a.encoder_state:encoder.restore(torch.load(a.encoder_state,map_location="cpu",weights_only=True))
    with torch.inference_mode(),(out/"manifest.jsonl").open("w") as stream:
        for r in records(a.manifest):
            packet=encoder.record(r)
            packet={k:v.cpu() if torch.is_tensor(v) else v for k,v in packet.items()}
            path=out/(hashlib.sha256(r["id"].encode()).hexdigest()+".pt")
            torch.save(packet,path)
            stream.write(json.dumps(dict(id=r["id"],episode_id=r["episode_id"],
                        feature_path=str(path.resolve()),encoder_revision=a.revision))+"\n")
            stream.flush()
if __name__=="__main__":main()
