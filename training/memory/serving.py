"""Opt-in native N0/LingBot serving hooks for a bound frozen Memory scorer.

These hooks change cache retention only. Native observation encoding, model
forward, attention arithmetic, action transforms and denoising stay upstream.
"""
import types
import torch
from n0_twam.models.global_kv_retention import token_rows
from .scorer_client import ScorerClient
from .serving_policy import FrozenMemoryRetention


def validate_config(cfg):
    if cfg.frame_chunk_size!=2 or cfg.action_per_frame!=12:
        raise ValueError("Serving must match native two-frame / twelve-action training")
    if cfg.height!=256 or cfg.width!=256 or len(cfg.obs_cam_keys)!=3:
        raise ValueError("Memory checkpoint requires three 256x256 cameras")
    if list(getattr(cfg,"tactile_keys",[])) or getattr(cfg,"env_type","none")!="none":
        raise ValueError("Memory is RGB-only with width-concatenated native cameras")
    if cfg.memory_scorer_sha256 is None:
        raise ValueError("An explicit frozen scorer binding is required")


class LingbotCache:
    """Use upstream attention intact; override only native slot allocation."""
    def __init__(self,server):
        self.server=server;self.model=server.transformer;self.context=None
        self.policy=FrozenMemoryRetention("cpu",server.memory_scorer.score)
        for block in self.model.blocks:
            attn=block.attn1
            cache=attn.attn_caches[server.cache_name]
            k=cache["k"]
            attn.init_kv_cache(server.cache_name,4096,k.shape[2],k.shape[3],k.device,k.dtype,k.shape[0])
            if not hasattr(attn,"memory_original_allocate"):
                attn.memory_original_allocate=attn.allocate_slots
                def allocate(a,cache_name,key_size):
                    owner=a.memory_owner
                    if owner.context is None:return a.memory_original_allocate(cache_name,key_size)
                    slots,victims=owner.context
                    if key_size!=len(slots):raise ValueError("Native attention token count drift")
                    c=a.attn_caches[cache_name]
                    c["mask"][victims.to(c["mask"].device)]=False
                    c["id"][victims.to(c["id"].device)]=-1
                    return slots.to(c["mask"].device)
                attn.allocate_slots=types.MethodType(allocate,attn)
            attn.memory_owner=self

    def forward(self,input_dict,update_cache=0,cache_name="pos",action_mode=False,train_mode=False):
        if train_mode:raise ValueError("Inference cache adapter cannot train")
        grid=input_dict["grid_id"].detach().cpu()
        rows=token_rows(dict(grid_id=grid),batch_size=grid.shape[0],length=grid.shape[-1],
            main_count=grid.shape[-1],action_mode=action_mode,update_cache=update_cache,device="cpu")
        caches=[b.attn1.attn_caches[cache_name] for b in self.model.blocks]
        oldmask=caches[0]["mask"].detach().cpu().clone()
        for cache in caches[1:]:
            if not torch.equal(cache["mask"].cpu(),oldmask) or not torch.equal(cache["id"].cpu(),caches[0]["id"].cpu()):
                raise ValueError("Layer cache masks diverged")
        slots,victims=self.policy.plan(oldmask,len(rows["kind"]),rows)
        changed=torch.cat((slots,victims)).unique()
        snapshots=[]
        for cache in caches:
            ix=changed.to(cache["k"].device)
            snapshots.append((cache,ix,{k:cache[k][:,ix].clone() for k in ("k","v")},
                {k:cache[k].clone() for k in ("mask","id","is_pred")}))
        previous=self.policy.snapshot()
        def restore():
            for cache,ix,kv,metadata in snapshots:
                for k,v in kv.items():cache[k][:,ix]=v
                for k,v in metadata.items():cache[k].copy_(v)
            self.policy.restore(previous)
        self.context=(slots,victims)
        try:
            result=self.model.memory_original_forward(input_dict,update_cache=update_cache,
                cache_name=cache_name,action_mode=action_mode,train_mode=False)
            if update_cache:self.policy.commit(slots,rows,oldmask)
            else:restore()  # temporary diffusion queries cannot evict history
            return result
        except BaseException:
            restore();raise
        finally:self.context=None


def install(Server,model):
    """Install before construction on a dedicated Memory server class."""
    if model not in ("n0twam","lingbot"):raise ValueError("Unknown Memory model")
    original_reset=Server._reset
    original_encode=Server._encode_obs
    original_infer=Server.infer
    def reset(self,prompt=None):
        validate_config(self.job_config)
        if not hasattr(self,"memory_scorer"):
            self.memory_scorer=ScorerClient(self.job_config.memory_scorer_endpoint,
                self.job_config.memory_scorer_sha256)
        self.memory_scorer.reset(prompt)
        original_reset(self,prompt)
        self.memory_failed=False
        if model=="n0twam":
            caches=self.transformer.mot.shared_attn
            cache=caches[0].attn_caches[self.cache_name]
            if len(cache["mask"])!=4096:raise ValueError("Memory requires exactly 4096 native KV slots")
            self.transformer.mot.retention_policies[self.cache_name]=FrozenMemoryRetention(cache["mask"].device,self.memory_scorer.score)
        else:
            if not hasattr(self.transformer,"memory_original_forward"):
                self.transformer.memory_original_forward=self.transformer.forward
            self.memory_cache=LingbotCache(self)
            self.transformer.forward=self.memory_cache.forward
    def encode(self,obs):
        result=original_encode(self,obs)
        if result is None:return result
        images=obs["obs"] if isinstance(obs["obs"],list) else [obs["obs"]]
        count=int(result.shape[2])
        cold=self.memory_scorer.next_frame==0
        anchors=[0]+list(range(4,len(images),4)) if cold else list(range(3,len(images),4))
        if len(anchors)!=count or len(images)!=(1+4*(count-1) if cold else 4*count):
            raise ValueError("Native streaming VAE anchors differ from training")
        if getattr(self.job_config,"memory_audit",False):
            self.memory_encoded_audit=(self.memory_scorer.next_frame,result.detach().cpu().clone())
        self.memory_scorer.append(images,self.job_config.obs_cam_keys,anchors)
        return result
    def infer(self,obs):
        if getattr(self,"memory_failed",False) and not obs.get("reset"):
            raise RuntimeError("Previous Memory request failed; reset the episode before continuing")
        try:return original_infer(self,obs)
        except BaseException:
            # LingBot's upstream streaming VAE has no transaction API. Abort the
            # episode on any error; never continue from partially grounded KV.
            self.memory_failed=True
            raise
    Server._reset=reset;Server._encode_obs=encode;Server.infer=infer
