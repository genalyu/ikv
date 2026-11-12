"""Canonical heuristic bridge: raw unit-weight inference, DINOv3, world steps."""
from dataclasses import dataclass
import hashlib, importlib, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .types import Candidate
from .scorer import ScorerBridge

def _module(source_root, name):
    root=Path(source_root).expanduser().resolve()
    if not (root/"n0_twam/models/heuristic_score.py").is_file():
        raise ValueError("Formula source must contain canonical heuristic_score.py")
    sys.path.insert(0,str(root))
    module=importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Formula dependency imported from a different checkout")
    return module

class DinoEncoder:
    def __init__(self,source_root,encoder_path,device):
        module=_module(source_root,"n0_twam.preprocessing.dinov3")
        self.model=module.FrozenDinoV3PatchEncoder.from_pretrained(
            encoder_path,device=device,torch_dtype=torch.float32,image_size=(224,224))
        self.device=device
        self.revision=getattr(self.model,"backbone_sha256",None)
        if not self.revision:raise ValueError("DINO requires a pinned hash descriptor")
    @torch.inference_mode()
    def image(self,rgb):
        rgb=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].to(self.device)
        return self.model(rgb).tokens[0].cpu()

@dataclass(frozen=True)
class FormulaFrame:
    time: float
    digest: str
    shape: tuple
    features: torch.Tensor
    contact_duration: torch.Tensor

class FormulaBridge:
    """Immutable observations, causal prototypes, no state publication on KV failure.

    Adapter timestamps remain seconds for learned-scoring compatibility. Only
    this backend converts them to world/control steps using world_time_hz.
    """
    def __init__(self,source_root,encoder_path=None,device="cuda:0",*,
                 world_time_hz,encoder=None,max_prototypes=4096):
        if not math.isfinite(world_time_hz) or world_time_hz<=0:raise ValueError("Invalid world clock")
        self.source_root,self.encoder_path,self.device=source_root,encoder_path,device
        self.world_time_hz=float(world_time_hz)
        self.max_prototypes=int(max_prototypes)
        if self.max_prototypes<=0:raise ValueError("Invalid prototype cap")
        self._heuristic=_module(source_root,"n0_twam.models.heuristic_score").heuristic_scores
        self.encoder=encoder or DinoEncoder(source_root,encoder_path,device)
        self.revision=self.encoder.revision
        self.reset()
    def reset(self):
        self._frames,self._frame_times={},{}
        self.encoded_images=self.score_calls=0
        self._score_cache=None
        self._prototype_cache=None
    def fork(self):
        return type(self)(self.source_root,self.encoder_path,self.device,
            world_time_hz=self.world_time_hz,encoder=self.encoder,max_prototypes=self.max_prototypes)
    @property
    def registered_frames(self):
        return tuple(sorted(self._frame_times,key=lambda f:(self._frame_times[f],f)))
    @torch.inference_mode()
    def register_frame(self,frame_id,timestamp,images,*,contact_duration=None):
        """Optional per-sensor scalar/DINO-grid contact durations, in world steps.

        Native VPP/HWM RGB contracts supply no contact sidecar; absent contact
        uses d=0. Supplied contact data is validated and never discarded.
        """
        timestamp=float(timestamp)
        if not isinstance(frame_id,str) or not frame_id or not math.isfinite(timestamp) or timestamp<0:
            raise ValueError("Invalid frame identity/time")
        if frame_id in self._frame_times and self._frame_times[frame_id]!=timestamp:
            raise ValueError("Frame changed timestamp")
        if not isinstance(images,dict) or not images:raise ValueError("images must map sensors")
        contacts={} if contact_duration is None else contact_duration
        if not isinstance(contacts,dict) or set(contacts)-set(images):raise ValueError("Contact sensors differ")
        pending={}
        for sensor,rgb in sorted(images.items()):
            rgb=np.asarray(rgb)
            if type(sensor) is not int or sensor<0 or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:
                raise ValueError("Expected sensor and uint8 RGB")
            key=(frame_id,sensor)
            digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            old=self._frames.get(key)
            contact=torch.as_tensor(contacts.get(sensor,0.),dtype=torch.float32)
            if not torch.isfinite(contact).all() or (contact<0).any():raise ValueError("Invalid contact duration")
            if old is not None:
                contact=torch.broadcast_to(contact,old.contact_duration.shape)
                if (old.time,old.digest,old.shape)!=(timestamp,digest,rgb.shape) or not torch.equal(old.contact_duration,contact):
                    raise ValueError("Frame changed RGB/contact metadata")
                continue
            features=self.encoder.image(rgb).detach().cpu().float().clone()
            if features.ndim!=3 or not torch.isfinite(features).all() or not all(features.shape):
                raise ValueError("DINO must emit finite H,W,D")
            contact=torch.broadcast_to(contact,features.shape[:2]).clone()
            pending[key]=FormulaFrame(timestamp,digest,rgb.shape,features,contact)
        self._frames.update(pending);self._frame_times[frame_id]=timestamp
        self.encoded_images+=len(pending)
        if pending:self._score_cache=None
    @staticmethod
    def _region_score(grid,box):
        y0,x0,y1,x1=box
        ey=torch.linspace(0,1,grid.shape[0]+1,dtype=torch.float64)
        ex=torch.linspace(0,1,grid.shape[1]+1,dtype=torch.float64)
        wy=(torch.minimum(ey[1:],torch.tensor(y1))-torch.maximum(ey[:-1],torch.tensor(y0))).clamp_min(0)
        wx=(torch.minimum(ex[1:],torch.tensor(x1))-torch.maximum(ex[:-1],torch.tensor(x0))).clamp_min(0)
        w=wy[:,None]*wx[None,:]
        return float((grid.double()*w).sum()/w.sum())
    def _grids(self,query_time):
        if self._score_cache is not None and self._score_cache[0]==query_time:return self._score_cache[1]
        keys=sorted((k for k,f in self._frames.items() if f.time<=query_time),
                    key=lambda k:(self._frames[k].time,k[0],k[1]))
        if not keys:raise ValueError("No causal observations")
        # Reuse an exact causal prefix; late native-window observations trigger
        # rebuild. Clone before update so errors cannot publish partial classes.
        width=self._frames[keys[0]].features.shape[-1]
        cache=self._prototype_cache
        if cache is not None and tuple(keys[:len(cache[0])])==cache[0]:
            start=len(cache[0]);count=cache[1]
            proto=cache[2].clone();last=cache[3].clone()
        else:
            start=count=0
            proto=torch.empty(self.max_prototypes,width)
            last=torch.empty(self.max_prototypes)
        for key in keys[start:]:
            frame=self._frames[key]
            values=F.normalize(frame.features.reshape(-1,frame.features.shape[-1]),dim=-1)
            if width!=values.shape[-1]:raise ValueError("DINO width changed")
            for value in values:
                if not value.ne(0).any():continue
                sims=proto[:count]@value
                best=int(sims.argmax()) if count else -1
                if best<0 or float(sims[best])<.75:
                    if count>=self.max_prototypes:raise RuntimeError("Formula prototype capacity reached")
                    best=count;count+=1;proto[best].copy_(value)
                last[best]=frame.time*self.world_time_hz
        self._prototype_cache=(tuple(keys),count,proto,last)
        proto=proto[:count];last=last[:count];grids={}
        for key in keys:
            frame=self._frames[key];shape=frame.features.shape[:2]
            features=frame.features.reshape(-1,frame.features.shape[-1])
            times=torch.full((len(features),),frame.time*self.world_time_hz)
            novelty=torch.zeros_like(times)
            if len(proto):
                rho,group=(F.normalize(features,dim=-1)@proto.T).max(-1)
                novelty=torch.exp(-(last[group]-times).clamp_min(0)/8.)*((rho>=.75)&features.ne(0).any(-1))
            grids[key]=self._heuristic(times,query_time*self.world_time_hz,
                torch.ones_like(times,dtype=torch.bool),frame.contact_duration.flatten(),
                novelty,weights=(1.,1.,1.)).reshape(shape)
        self._score_cache=(query_time,grids)
        return grids
    @torch.inference_mode()
    def score(self,candidates,task,query_time,subtask="",memory=""):
        candidates=tuple(candidates)
        if not candidates:return {}
        query_time=float(query_time)
        if not math.isfinite(query_time) or query_time<0:raise ValueError("Invalid query time")
        if any(not isinstance(c,Candidate) for c in candidates):raise TypeError("Expected Candidate")
        if len({c.uid for c in candidates})!=len(candidates):raise ValueError("Duplicate candidate")
        grids=self._grids(query_time);output={}
        for c in candidates:
            if c.time>query_time:raise ValueError("Future candidate")
            for r in c.regions:
                key=(r.frame_id,r.sensor)
                if key not in grids or self._frames[key].time>c.time+1e-6:raise ValueError("Missing/future provenance")
            output[c.uid]=sum(r.weight*self._region_score(grids[(r.frame_id,r.sensor)],r.box)
                for r in c.regions)/sum(r.weight for r in c.regions)
        self.score_calls+=1
        return output

def make_scorer(config,*,world_time_hz):
    backend=config.get("backend","learned")
    if backend=="formula":
        if config.get("planner_mode")=="interactive":
            raise ValueError("Formula has no Planner-feedback input; use lower_only")
        return FormulaBridge(config["source_root"],config["encoder_path"],config.get("device","cuda:0"),
            world_time_hz=world_time_hz,max_prototypes=config.get("max_prototypes",4096))
    if backend!="learned":raise ValueError("Unknown IKV backend")
    return ScorerBridge(config["source_root"],config["checkpoint"],config["encoder_path"],config.get("device","cuda:0"))
