"""Average fully accumulated gradients after reentrant backward has finished."""
import torch
import torch.distributed as dist
@torch.no_grad()
def average_gradients(parameters,bucket_bytes=32*1024*1024):
 if not dist.is_initialized() or dist.get_world_size()==1:return
 params=[p for p in parameters if p.requires_grad]
 if not params:return
 # All ranks must have the same complete parameter coverage before collectives.
 valid=torch.tensor(int(all(p.grad is not None and not p.grad.is_sparse for p in params)),device=params[0].device)
 dist.all_reduce(valid,op=dist.ReduceOp.MIN)
 if not valid.item():raise RuntimeError("Missing or sparse trainable gradients on at least one rank")
 world=dist.get_world_size();bucket=[];size=0
 def flush():
  if not bucket:return
  flat=torch.cat([g.detach().reshape(-1).float() for g in bucket])
  dist.all_reduce(flat,op=dist.ReduceOp.SUM);flat.div_(world)
  offset=0
  for g in bucket:
   n=g.numel();g.copy_(flat[offset:offset+n].view_as(g));offset+=n
 for p in params:
  g=p.grad
  # Split large tensors, keeping the FP32 communication temporary bounded.
  for chunk in g.view(-1).split(max(1,bucket_bytes//4)):
   n=chunk.numel()*4
   if bucket and size+n>bucket_bytes:flush();bucket=[];size=0
   bucket.append(chunk);size+=n
 flush()
