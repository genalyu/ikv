"""Inactive candidate: preserve2048-token GEMM shapes; avoid redundant INNER remat only on short GPU-resident inputs."""
import torch,contextlib
def enable_mlp_without_inner_remat(model,max_tokens=100000,chunk_tokens=2048):
 count=0
 for module in model.modules():
  if type(module).__name__!="Qwen3_5MLP" or getattr(module,"_no_inner_chunk_checkpoint",False):continue
  if not getattr(module,"_token_chunked",False):raise RuntimeError("Existing chunked MLP required")
  chunked=module.forward;original=chunked.__defaults__[0]
  assert original.__self__ is module
  def forward(x,_original=original,_chunked=chunked):
   if x.shape[-2]>max_tokens:return _chunked(x)
   if x.shape[-2]<=chunk_tokens:return _original(x)
   outputs=[]
   for start in range(0,x.shape[-2],chunk_tokens):
    # Each chunk must retain its own FP32->BF16 parameter cast node.
    # A shared autocast weight cache would sum chunk gradients in BF16 first.
    ctx=torch.autocast("cuda",dtype=torch.get_autocast_dtype("cuda"),enabled=torch.is_autocast_enabled(),cache_enabled=False) if x.device.type=="cuda" else contextlib.nullcontext()
    with ctx:outputs.append(_original(x[...,start:start+chunk_tokens,:]))
   return torch.cat(outputs,dim=-2)
  module.forward=forward;module._no_inner_chunk_checkpoint=True;count+=1
 return count
