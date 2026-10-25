"""Feature packets and the existing RGB teacher-label contract."""
import json
from pathlib import Path
import torch
import numpy as np
from training.qwen_scorer.common import decode_scores

FIELDS=("features","text","times","xy","modalities","sensors")
def arguments(packet,device):
    return {k:packet[k].to(device) for k in FIELDS} | {"query_time":float(packet["query_time"])}

def validate_packet(p):
    for k in (*FIELDS,"query_time","ids","encoder_revision"):
        if k not in p:raise ValueError("Missing packet field: "+k)
    n=len(p["features"])
    if len(p["ids"])!=n or len(set(p["ids"]))!=n:
        raise ValueError("Candidate IDs must be unique and aligned")
    if not p["encoder_revision"]:raise ValueError("Immutable encoder revision required")
    if "targets" in p:
        y=p["targets"]
        if y.shape!=(n,) or not torch.isfinite(y).all() or ((y<0)|(y>100)).any():
            raise ValueError("Targets must be finite scores in [0,100]")
    if "supervised" in p and p["supervised"].shape!=(n,):
        raise ValueError("Supervision mask mismatch")
    if (p["times"]>p["query_time"]).any():raise ValueError("Future features")

def load_packet(path,revision=None):
    p=torch.load(path,map_location="cpu",weights_only=True);validate_packet(p)
    if revision is not None and p["encoder_revision"]!=revision:
        raise ValueError("Stale/incompatible encoder feature cache")
    return p

def records(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:raise ValueError("Empty manifest")
    return rows

def targets(record):
    y=decode_scores(json.loads(record["target"]),int(record["query_frame"])+1)
    return torch.from_numpy(y.copy()).flatten().float()

def regression_loss(logits,y,mask=None,rank_weight=.1,pairs=4096):
    mask=torch.ones_like(y,dtype=torch.bool) if mask is None else mask.bool()
    if not mask.any():raise ValueError("No supervised targets")
    z=logits[mask].float();y=y[mask].float()/100.
    # Each query has equal weight irrespective of prefix length.
    point=torch.nn.functional.smooth_l1_loss(z.sigmoid(),y)
    a=torch.randint(len(z),(pairs,),device=z.device)
    b=torch.randint(len(z),(pairs,),device=z.device)
    sign=torch.sign(y[a]-y[b]);valid=sign!=0
    rank=(torch.nn.functional.softplus(-sign[valid]*(z[a[valid]]-z[b[valid]])).mean()
          if valid.any() else z.sum()*0.)
    return point+rank_weight*rank

def check_disjoint(train,validation):
    a={r["episode_id"] for r in train};b={r["episode_id"] for r in validation}
    if a & b:raise ValueError("Episode leakage between train and validation")


def load_record(record,revision=None):
    if "episode_features" in record:
        from .episode_cache import episode_packet
        return episode_packet(record,revision)
    return load_packet(record["feature_path"],revision)
