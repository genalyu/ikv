"""Prepare native-anchor Qwen features before selecting a scorer checkpoint."""
import argparse,json,time
from pathlib import Path
from .sidecar import verify_encoder,sha256
from .feature_cache import episode_features,CANONICAL_IMAGE_BATCH

def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True)
    p.add_argument("--encoder",required=True);p.add_argument("--encoder-manifest",required=True)
    p.add_argument("--output",required=True);p.add_argument("--task",required=True)
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args();campaign=json.loads(Path(a.campaign).read_text())
    revision=json.loads(Path(a.encoder_manifest).read_text())["sha"]+":linear-patch-v1"
    hashes=verify_encoder(a.encoder,revision,a.encoder_manifest)
    parts=json.loads(Path(campaign["splits"]).read_text())["per_task"][a.task]
    root=Path(campaign["prepared_root"])/a.task/"dataset"
    episodes=[json.loads(x) for x in (root/"meta/episodes.jsonl").read_text().splitlines()]
    prompt=json.loads((Path(campaign["task_config_root"])/f"{a.task}.json").read_text())["prompt"]
    encoder=None
    def get_encoder():
        nonlocal encoder
        if encoder is None:
            from training.parallel_scorer.encoders import QwenEncoders
            encoder=QwenEncoders(a.encoder,revision,a.device,image_batch=CANONICAL_IMAGE_BATCH).eval()
        return encoder
    completed=[]
    for ep in sorted(parts["train"]+parts["validation"]):
        row=next(x for x in episodes if x["episode_index"]==ep)
        *_,key=episode_features(root,ep,row["length"],campaign["cameras"],prompt,revision,hashes,get_encoder,a.output)
        completed.append(dict(episode=ep,key=key));print(json.dumps(completed[-1]),flush=True)
    marker=Path(a.output)/f"READY-{a.task}.json"
    marker.write_text(json.dumps(dict(status="passed",task=a.task,encoder_revision=revision,encoder_image_batch=CANONICAL_IMAGE_BATCH,episodes=completed,
        campaign_sha256=sha256(a.campaign),completed=time.time()),indent=2))
if __name__=="__main__":main()
