"""Exact all-history attention. Query tiling never truncates the key set."""
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ScorerConfig:
    vision_dim: int = 1024
    text_dim: int = 1024
    width: int = 512
    layers: int = 4
    heads: int = 8
    query_chunk: int = 2048
    ff_chunk: int = 2048
    sensors: int = 32
    checkpoint_layers: bool = True
    flash_only: bool = False

    @classmethod
    def preset(cls, name, **kwargs):
        presets = {
            "p067m": dict(width=64, layers=1, heads=1),
            "p26m": dict(width=128, layers=1, heads=2),
            "p59m": dict(width=192, layers=1, heads=3),
            "p1m": dict(width=256, layers=1, heads=4),
            "p5m": dict(width=384, layers=2, heads=6),
            "p8m": dict(width=512, layers=2, heads=8),
            "p13m": dict(width=512, layers=3, heads=8),
            "small": dict(width=512, layers=4, heads=8),
            "medium": dict(width=768, layers=6, heads=12),
        }
        if name not in presets:
            raise ValueError("Unknown scorer preset: " + name)
        return cls(**dict(presets[name], **kwargs))


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Linear(c.width, c.width)
        self.kv = nn.Linear(c.width, 2*c.width)
        self.out = nn.Linear(c.width, c.width)
        self.heads, self.chunk, self.flash_only = c.heads, c.query_chunk, c.flash_only

    def forward(self, query, context):
        n, d = query.shape
        q = self.q(query).reshape(n,self.heads,-1).transpose(0,1).unsqueeze(0)
        k,v = self.kv(context).reshape(-1,2,self.heads,d//self.heads).unbind(1)
        k,v = [a.transpose(0,1).unsqueeze(0) for a in (k,v)]
        pieces=[]
        # ALL keys/values remain visible for every query tile. No local window.
        for begin in range(0,n,self.chunk):
            if self.flash_only:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                scope=sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            else:
                scope=nullcontext()
            with scope:
                z=F.scaled_dot_product_attention(q[:,:,begin:begin+self.chunk],k,v,
                                                 dropout_p=0.,is_causal=False)
            pieces.append(z.squeeze(0).transpose(0,1).reshape(-1,d))
        return self.out(torch.cat(pieces,dim=0))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.self_attn=Attention(c); self.task_attn=Attention(c)
        self.norm1=nn.LayerNorm(c.width);self.norm2=nn.LayerNorm(c.width)
        self.norm3=nn.LayerNorm(c.width)
        self.ff=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),
                              nn.Linear(4*c.width,c.width))
        self.chunk=c.ff_chunk
    def forward(self,x,text):
        h=self.norm1(x);x=x+self.self_attn(h,h)
        x=x+self.task_attn(self.norm2(x),text)
        return torch.cat([v+self.ff(self.norm3(v)) for v in x.split(self.chunk)],dim=0)

class ParallelScorer(nn.Module):
    def __init__(self,c):
        super().__init__()
        if c.width%c.heads or min(c.layers,c.query_chunk,c.ff_chunk,c.sensors)<1:
            raise ValueError("Invalid scorer configuration")
        self.config=c
        self.vision=nn.Linear(c.vision_dim,c.width)
        self.text=nn.Sequential(nn.Linear(c.text_dim,c.width),nn.LayerNorm(c.width))
        # metadata: age, row, col, current flag; no list-index positional embedding
        self.metadata=nn.Sequential(nn.Linear(4,c.width),nn.SiLU(),nn.Linear(c.width,c.width))
        self.modality=nn.Embedding(2,c.width);self.sensor=nn.Embedding(c.sensors,c.width)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.head=nn.Sequential(nn.LayerNorm(c.width),nn.Linear(c.width,c.width//2),
                                 nn.GELU(),nn.Linear(c.width//2,1))

    def forward(self, features, text, times, xy, modalities, sensors, query_time):
        n=len(features)
        if n==0 or len(text)==0: raise ValueError("Nonempty history and task required")
        if features.ndim!=2 or text.ndim!=2 or xy.shape!=(n,2):
            raise ValueError("Expected features [N,D], text [L,D], xy [N,2]")
        if any(t.shape!=(n,) for t in (times,modalities,sensors)):
            raise ValueError("Metadata length mismatch")
        if (times>query_time).any(): raise ValueError("Future observation in history")
        if ((modalities<0)|(modalities>1)).any(): raise ValueError("modality: visual=0/tactile=1")
        if ((sensors<0)|(sensors>=self.config.sensors)).any(): raise ValueError("Invalid sensor")
        if not all(torch.isfinite(t).all() for t in (features,text,times,xy)):
            raise ValueError("Nonfinite input")
        age=torch.log1p((query_time-times).float())
        meta=torch.cat([age[:,None],xy.float(),(times==query_time).float()[:,None]],dim=1)
        if features.device.type == "cpu":
            features=features.to(self.vision.weight.dtype);text=text.to(self.text[0].weight.dtype)
        x=self.vision(features)+self.metadata(meta.to(self.metadata[0].weight.dtype))
        x=x+self.modality(modalities.long())+self.sensor(sensors.long())
        task=self.text(text)
        for block in self.blocks:
            if self.training and self.config.checkpoint_layers:
                x=checkpoint(block,x,task,use_reentrant=False)
            else:x=block(x,task)
        return self.head(x).squeeze(-1)

    def counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    interaction=sum(p.numel() for p in self.blocks.parameters()))

def select_indices(logits, modalities, budgets, eligible=None, protected=None):
    """Budgets INCLUDE protected candidates. Actions are outside this API.
    Input order is stable candidate-ID order for deterministic ties.
    """
    n=len(logits)
    eligible=torch.ones(n,dtype=torch.bool,device=logits.device) if eligible is None else eligible.bool()
    protected=torch.zeros_like(eligible) if protected is None else protected.bool()
    if eligible.shape!=(n,) or protected.shape!=(n,) or modalities.shape!=(n,):
        raise ValueError("Mask shape mismatch")
    if (protected & ~eligible).any() or not torch.isfinite(logits).all():
        raise ValueError("Invalid protection or nonfinite score")
    if ((modalities<0)|(modalities>1)).any():raise ValueError("Invalid modality")
    chosen=[]
    for m in (0,1):
        b=int(budgets[m])
        fixed=torch.where(protected & (modalities==m))[0]
        if b<0 or len(fixed)>b:raise ValueError("Budget smaller than protected set")
        pool=torch.where(eligible & ~protected & (modalities==m))[0]
        order=torch.argsort(logits[pool],descending=True,stable=True)
        chosen.extend([fixed,pool[order[:max(0,b-len(fixed))]]])
    return torch.cat(chosen).sort().values

def save_scorer(model,path,**extra):
    torch.save(dict(config=asdict(model.config),state=model.state_dict(),**extra),path)

def load_scorer(path,device="cpu"):
    obj=torch.load(path,map_location=device,weights_only=True)
    model=ParallelScorer(ScorerConfig(**obj["config"])).to(device)
    model.load_state_dict(obj["state"])
    return model,obj
