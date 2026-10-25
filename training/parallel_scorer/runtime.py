"""Reusable streaming feature store and KV-ID selection adapter.
Only context-independent image features are cached, never contextual scores.
"""
from pathlib import Path
import torch
from .model import select_indices,load_scorer
from .data import arguments,validate_packet

class FeatureBank:
    def __init__(self,revision):
        if not revision:raise ValueError("Encoder revision is required")
        self.revision=revision;self.rows={};self.dimension=None
    def append(self,ids,features,times,xy,modalities,sensors,revision):
        if revision!=self.revision:raise ValueError("Encoder revision changed; rebuild bank")
        if features.ndim!=2:raise ValueError("Features must be [N,D]")
        n=len(ids)
        if len(set(ids))!=n or any(i in self.rows for i in ids):raise ValueError("Duplicate candidate IDs")
        if len(features)!=n or xy.shape!=(n,2) or any(v.shape!=(n,) for v in (times,modalities,sensors)):
            raise ValueError("Feature metadata mismatch")
        if self.dimension is not None and features.shape[1]!=self.dimension:raise ValueError("Feature dimension changed")
        if not all(torch.isfinite(v).all() for v in (features,times,xy)):raise ValueError("Nonfinite features")
        if ((modalities<0)|(modalities>1)).any():raise ValueError("Invalid modality")
        self.dimension=features.shape[1]
        tensors=[v.detach().cpu().clone() for v in (features,times,xy,modalities,sensors)]
        for j,i in enumerate(ids):self.rows[i]=tuple(v[j] for v in tensors)
    def packet(self,text,query_time):
        # Stable IDs, not insertion order, break exact score ties.
        ids=sorted(self.rows)
        if not ids:raise ValueError("Empty feature bank")
        values=[torch.stack([self.rows[i][k] for i in ids]) for k in range(5)]
        return dict(zip(("features","times","xy","modalities","sensors"),values),
                    text=text,ids=ids,query_time=query_time,encoder_revision=self.revision)
    def save(self,path):
        torch.save(dict(revision=self.revision,dimension=self.dimension,rows=self.rows),path)
    @classmethod
    def load(cls,path,revision):
        obj=torch.load(path,map_location="cpu",weights_only=True)
        if obj["revision"]!=revision:raise ValueError("Stale feature bank")
        bank=cls(revision);bank.dimension=obj["dimension"];bank.rows=obj["rows"];return bank

class RuntimeScorer:
    def __init__(self,checkpoint,device="cuda:0"):
        self.model,self.info=load_scorer(checkpoint,device);self.model.eval();self.device=device
        if str(device).startswith("cuda"):self.model.to(torch.bfloat16)
    @torch.inference_mode()
    def score(self,packet,budgets,eligible=None,protected=None,kv_candidate_ids=None,kv_modalities=None):
        validate_packet(packet)
        if packet["encoder_revision"]!=self.info["encoder_revision"]:raise ValueError("Encoder/checkpoint mismatch")
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=self.device.startswith("cuda")):
            scores=self.model(**arguments(packet,self.device)).float()
        ids=packet["ids"];mods=packet["modalities"].to(self.device)
        # Explicit map may have multiple native KV slots per scoring region.
        # Budget is applied AFTER mapping, in native KV slots, not image regions.
        if kv_candidate_ids is not None:
            lookup={i:k for k,i in enumerate(ids)}
            if any(i not in lookup for i in kv_candidate_ids):raise ValueError("Unmapped native KV")
            index=torch.tensor([lookup[i] for i in kv_candidate_ids],device=self.device)
            scores=scores[index];mods=mods[index];ids=list(kv_candidate_ids)
            if kv_modalities is not None and not torch.equal(mods.cpu(),torch.as_tensor(kv_modalities)):
                raise ValueError("Native/scorer modality mismatch")
        move=lambda v:None if v is None else torch.as_tensor(v,device=self.device)
        selected=select_indices(scores,mods,budgets,move(eligible),move(protected))
        return dict(scores=scores.cpu(),selected_indices=selected.cpu(),
                    selected_candidate_ids=[ids[k] for k in selected.tolist()])
