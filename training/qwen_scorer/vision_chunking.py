import json,os,time
import torch
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPooling

def enable_chunked_vision(visual,images_per_chunk=6,deterministic_attention=False):
 if getattr(visual,"_ikv_chunked",False):return
 if images_per_chunk<1:raise ValueError("images_per_chunk must be positive")
 original=visual.forward
 # One checkpoint surrounds each independent image batch; avoid nested per-block checkpoints.
 for module in visual.modules():
  if hasattr(module,"gradient_checkpointing"):module.gradient_checkpointing=False
 def forward(hidden_states,grid_thw,**kwargs):
  if deterministic_attention:kwargs["deterministic"]=True
  sizes=grid_thw.prod(-1).tolist()
  assert sum(sizes)==hidden_states.shape[0]
  if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
   raise ValueError("Chunked vision feature path does not expose internal hidden states")
  merged=[];offset=0
  for start in range(0,len(sizes),images_per_chunk):
   end=min(len(sizes),start+images_per_chunk);length=sum(sizes[start:end])
   pixels=hidden_states[offset:offset+length];grid=grid_thw[start:end];offset+=length
   # Bind grid and kwargs per chunk so backward recomputation uses the same images.
   def encode(x,chunk_grid=grid,chunk_kwargs=dict(kwargs)):
    return original(x,grid_thw=chunk_grid,**chunk_kwargs).pooler_output
   features=checkpoint(encode,pixels,use_reentrant=False) if torch.is_grad_enabled() and visual.training else encode(pixels)
   merged.append(features)
   if start==0 or end==len(sizes) or end%192==0:
    print(json.dumps(dict(event="vision_chunk_progress",rank=os.environ.get("RANK"),images_done=end,images_total=len(sizes),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,time=time.time())),flush=True)
  assert offset==hidden_states.shape[0]
  return BaseModelOutputWithPooling(pooler_output=torch.cat(merged,dim=0))
 visual.forward=forward;visual._ikv_chunked=True
