"""Token-independent MLP chunking; checkpoint chunks to bound temporary activations."""
import torch,contextlib
from torch.utils.checkpoint import checkpoint
def enable_chunked_mlp(model,chunk_tokens=2048,gpu_recompute=False):
 count=0
 for module in model.modules():
  if type(module).__name__!="Qwen3_5MLP" or getattr(module,"_token_chunked",False):continue
  original=module.forward
  def forward(x,_original=original):
   def compute(z):
    with torch.autograd.graph.saved_tensors_hooks(lambda t:t,lambda t:t) if gpu_recompute and torch.is_grad_enabled() else contextlib.nullcontext():
     return _original(z)
   if x.shape[-2]<=chunk_tokens:return _original(x)
   outputs=[]
   for start in range(0,x.shape[-2],chunk_tokens):
    part=x[...,start:start+chunk_tokens,:]
    outputs.append(checkpoint(compute,part,use_reentrant=True) if torch.is_grad_enabled() and part.requires_grad else _original(part))
   return torch.cat(outputs,dim=-2)
  module.forward=forward;module._token_chunked=True;count+=1
 return count
