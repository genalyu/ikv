"""Qwen3.5 visual + task-only language encoding; no history language pass.
Use a pinned, local checkpoint. Trainable visual encoding must not use cached packets.
"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import numpy as np
from PIL import Image
from .data import targets,validate_packet

class QwenEncoders(nn.Module):
    def __init__(self,path,revision,device="cuda:0",train_vision=False,train_text=False,
                 image_batch=8,linear_patch=True):
        super().__init__()
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        if not revision:raise ValueError("Pinned encoder revision required")
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        base=Qwen3_5ForConditionalGeneration.from_pretrained(path,local_files_only=True,
               dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
        self.core=base.model.to(device)  # drop output vocabulary head
        if linear_patch:
            from training.qwen_scorer.vision_patch_linear import enable_linear_patch_embed
            enable_linear_patch_embed(self.core.visual)
        self.core.requires_grad_(False)
        self.core.visual.requires_grad_(train_vision)
        self.core.language_model.requires_grad_(train_text)
        self.revision=revision;self.image_batch=image_batch
        self.train_vision=train_vision;self.train_text=train_text
        self.vision_dim=base.config.text_config.hidden_size
        self.text_dim=base.config.text_config.hidden_size
        self._task_cache={}
        self.eval()

    def task(self,text):
        device=next(self.core.parameters()).device
        if not self.train_text and text in self._task_cache:
            return self._task_cache[text].to(device).clone()
        tokens=self.processor.tokenizer(text,return_tensors="pt",add_special_tokens=True)
        tokens={k:v.to(device) for k,v in tokens.items()}
        with torch.set_grad_enabled(self.training and self.train_text):
            out=self.core.language_model(**tokens,use_cache=False,return_dict=True).last_hidden_state[0]
        if not self.train_text:self._task_cache[text]=out.detach().cpu()
        return out

    def images(self,images):
        device=next(self.core.parameters()).device
        outputs=[]
        for begin in range(0,len(images),self.image_batch):
            batch=self.processor.image_processor(images=images[begin:begin+self.image_batch],
                  return_tensors="pt",size={"shortest_edge":65536,"longest_edge":65536})
            grid=batch["image_grid_thw"].to(device)
            if not torch.equal(grid,torch.tensor([[1,16,16]]*len(grid),device=device)):
                raise ValueError("Teacher grid requires 256x256, native 16x16 -> merged 8x8")
            pixels=batch["pixel_values"].to(device)
            def encode(p,g):
                o=self.core.get_image_features(p,g,return_dict=True).pooler_output
                return torch.cat(o,dim=0) if isinstance(o,(tuple,list)) else o
            with torch.set_grad_enabled(self.training and self.train_vision):
                if self.training and self.train_vision:
                    f=checkpoint(encode,pixels,grid,use_reentrant=False)
                else:f=encode(pixels,grid)
            if f.shape[0]!=len(grid)*64:raise ValueError("Unexpected native merger layout")
            outputs.append(f)
        return torch.cat(outputs)

    def record(self,r):
        # Official existing teacher data: three RGB cameras, full prefix.
        array=np.load(r["rgb_npy"],mmap_mode="r");n=int(r["query_frame"])+1
        if array.ndim!=5 or array.shape[1:]!=(3,256,256,3) or len(array)<n:
            raise ValueError("Expected clean RGB [frames,3,256,256,3]")
        images=[Image.fromarray(array[f,c]) for f in range(n) for c in range(3)]
        features=self.images(images)
        count=n*192;device=features.device
        f=torch.arange(n).repeat_interleave(192)
        sensor=torch.arange(3).repeat_interleave(64).repeat(n)
        xy=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing="ij"),-1).reshape(-1,2)/7.
        p=dict(features=features,text=self.task(r["task"]),
               times=f.float().to(device)/float(r["fps"]),xy=xy.repeat(n*3,1).to(device),
               modalities=torch.zeros(count,dtype=torch.long,device=device),
               sensors=sensor.to(device),query_time=(n-1)/float(r["fps"]),
               ids=[f'{r["episode_id"]}:f{frame}:c{cam}:p{patch}'
                    for frame in range(n) for cam in range(3) for patch in range(64)],
               encoder_revision=self.revision)
        if "target" in r:p["targets"]=targets(r).to(device)
        validate_packet(p);return p

    def export_trainable(self):
        # Save only changed components; frozen language weights stay in pinned base.
        result={}
        if self.train_vision:result["visual"]=self.core.visual.state_dict()
        if self.train_text:result["language_model"]=self.core.language_model.state_dict()
        return result

    def restore(self,state):
        for key,weights in state.items():
            getattr(self.core,key).load_state_dict(weights,strict=True)
        self._task_cache.clear()
