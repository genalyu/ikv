"""Bind a frozen scorer and prepare causal train/validation KV plans.

Usage: python -m training.memory.prepare_plans --campaign CONFIG --checkpoint
SCORER.pt --encoder QWEN_DIRECTORY --output PLAN_DIRECTORY. Test episodes are
never consumed here. A COMPLETE marker appears only after every required plan.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from .selection import VERSION, replay, scorer_to_native
from .sidecar import sha256, verify_encoder
from .feature_cache import episode_features, extract, CANONICAL_IMAGE_BATCH
from training.parallel_scorer.encoders import QwenEncoders
from training.parallel_scorer.runtime import RuntimeScorer


def write_json(path, data):
    path=Path(path);tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)



def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--encoder",required=True)
    p.add_argument("--encoder-manifest",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--feature-root",default=None)
    a=p.parse_args();campaign=json.loads(Path(a.campaign).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    scorer=RuntimeScorer(a.checkpoint,a.device)
    revision=scorer.info["encoder_revision"]
    if not a.device.startswith("cuda"):raise ValueError("Audited native plans require CUDA BF16 scoring")
    if revision.split(":")[1:]!=["linear-patch-v1"]:raise ValueError("Only the frozen linear-patch-v1 encoder is supported")
    encoder_hashes=verify_encoder(a.encoder,revision,a.encoder_manifest)
    input_hashes={"splits":sha256(campaign["splits"])}
    for task in json.loads(Path(campaign["splits"]).read_text())["per_task"]:
        input_hashes[task]=sha256(Path(campaign["task_config_root"])/f"{task}.json")
    binding=dict(version=VERSION,chunk_size=2,encoder_sha256=encoder_hashes,inputs_sha256=input_hashes,checkpoint=str(Path(a.checkpoint).absolute()),
        scorer_sha256=sha256(a.checkpoint),encoder=str(Path(a.encoder).absolute()),
        encoder_revision=revision,encoder_image_batch=CANONICAL_IMAGE_BATCH,budgets=[3584,512],plan_root=str(out.absolute()),
        campaign_sha256=sha256(a.campaign),prefix_sampling="native-VAE-anchor-RGB",
        rgb_contract="canonical-RGB-torch-bilinear-u8-round-v1")
    bp=out/"binding.json"
    if bp.exists() and json.loads(bp.read_text())!=binding:
        raise ValueError("Plan output belongs to a different scorer/campaign; choose a new directory")
    write_json(bp,binding)
    encoder=None
    def get_encoder():
        nonlocal encoder
        if encoder is None:encoder=QwenEncoders(a.encoder,revision,a.device,image_batch=CANONICAL_IMAGE_BATCH).eval()
        return encoder
    split=json.loads(Path(campaign["splits"]).read_text())["per_task"]
    cameras=campaign["cameras"]
    done=[]
    for task,selection in split.items():
        root=Path(campaign["prepared_root"])/task/"dataset"
        conversion=root/"conversion.json"
        if not conversion.exists():raise ValueError(f"Conversion incomplete: {task}")
        episodes=[json.loads(x) for x in (root/"meta/episodes.jsonl").read_text().splitlines()]
        prompt=json.loads((Path(campaign["task_config_root"])/f"{task}.json").read_text())["prompt"]
        ids=sorted(selection["train"]+selection["validation"])
        for episode in ids:
            row=next(v for v in episodes if v["episode_index"]==episode)
            length=int(row["length"]);name=f"episode_{episode:06d}_0_{length}.pt"
            target=out/task/name;target.parent.mkdir(exist_ok=True)
            if target.exists():
                old=torch.load(target,map_location="cpu",weights_only=True)
                if old["binding_sha256"]!=sha256(bp):raise ValueError("Stale completed plan")
                done.append(str(target));continue
            features,text,times,anchors,sources,feature_key=episode_features(
                root,episode,length,cameras,prompt,revision,encoder_hashes,get_encoder,a.feature_root)
            frames=len(features)
            xy=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing="ij"),-1).reshape(-1,2)/7.
            def score_prefix(n):
                packet=dict(features=features[:n].reshape(n*192,-1),text=text,
                    times=times[:n].repeat_interleave(192),xy=xy.repeat(n*3,1),
                    modalities=torch.zeros(n*192,dtype=torch.long),
                    sensors=torch.arange(3).repeat_interleave(64).repeat(n),
                    query_time=float(times[n-1]),ids=list(range(n*192)),encoder_revision=revision)
                scores=scorer.score(packet,{0:0,1:0})["scores"]
                return scorer_to_native(scores.reshape(n,3,8,8)).flatten()
            selected=replay(frames,score_prefix)
            data=dict(version=VERSION,chunk_size=2,selected=selected,frame_ids=anchors,cameras=cameras,
                budgets=binding["budgets"],scorer_sha256=binding["scorer_sha256"],
                binding_sha256=sha256(bp),latent_sha256=sources,feature_cache_key=feature_key)
            tmp=target.with_suffix(".tmp");torch.save(data,tmp);tmp.replace(target)
            done.append(str(target))
            print(json.dumps(dict(task=task,episode=episode,frames=frames,completed=len(done))),flush=True)
    write_json(out/"COMPLETE.json",dict(binding_sha256=sha256(bp),plans=done,completed=time.time()))

if __name__=="__main__":main()
