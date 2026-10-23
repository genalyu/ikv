"""Memory-bounded segmented delta backward, preserving gradients through every carry."""
import torch,contextlib
GPU_SEGMENT_RECOMPUTE=False
from torch.autograd.function import once_differentiable

class SegmentedDelta(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta,initial_state,kernel,segment_tokens,kwargs):
  ctx.gpu_segment_recompute=GPU_SEGMENT_RECOMPUTE
  ctx.kernel=kernel;ctx.segment_tokens=segment_tokens;ctx.kwargs=kwargs
  ctx.has_initial=initial_state is not None
  state=initial_state;states=[];outputs=[]
  for start in range(0,q.shape[1],segment_tokens):
   states.append(state)
   end=min(q.shape[1],start+segment_tokens)
   out,state=kernel(q[:,start:end].contiguous(),k[:,start:end].contiguous(),v[:,start:end].contiguous(),g=g[:,start:end].contiguous(),beta=beta[:,start:end].contiguous(),initial_state=state,output_final_state=True,**kwargs)
   if not torch.isfinite(out).all() or not torch.isfinite(state).all():
    raise FloatingPointError(f"Nonfinite segmented delta tokens [{start},{end})")
   outputs.append(out)
  ctx.save_for_backward(q,k,v,g,beta,*states)
  return torch.cat(outputs,dim=1),state
 @staticmethod
 @once_differentiable
 def backward(ctx,do,dstate):
  saved=ctx.saved_tensors
  inputs=saved[:5];states=saved[5:]
  grads=[torch.empty_like(x) for x in inputs]
  # Only one segment graph is alive. SliceBackward must never allocate a
  # full-sequence zero gradient for every segment of each q/k/v input.
  for i in range(len(states)-1,-1,-1):
   start=i*ctx.segment_tokens;end=min(inputs[0].shape[1],start+ctx.segment_tokens)
   with torch.enable_grad(), (torch.autograd.graph.saved_tensors_hooks(lambda t:t,lambda t:t) if ctx.gpu_segment_recompute else contextlib.nullcontext()):
    xs=[x[:,start:end].detach().contiguous().requires_grad_(True) for x in inputs]
    state=states[i]
    h=state.detach().requires_grad_(True) if state is not None else None
    out,ht=ctx.kernel(xs[0],xs[1],xs[2],g=xs[3],beta=xs[4],initial_state=h,output_final_state=True,**ctx.kwargs)
    targets=xs+([h] if h is not None else [])
    if dstate is None:
     gs=torch.autograd.grad(out,targets,do[:,start:end].contiguous())
    else:
     gs=torch.autograd.grad((out,ht),targets,(do[:,start:end].contiguous(),dstate))
   for dst,grad in zip(grads,gs[:5]):
    if not torch.isfinite(grad).all():raise FloatingPointError(f"Nonfinite segmented delta gradient tokens [{start},{end})")
    dst[:,start:end].copy_(grad)
   dstate=gs[5] if h is not None else None
   if dstate is not None and not torch.isfinite(dstate).all():raise FloatingPointError("Nonfinite carry gradient")
   del out,ht,gs,xs,targets,h
  return (*grads,dstate if ctx.has_initial else None,None,None,None)

def bounded_segmented_call(kernel,q,k,v,g,beta,*,segment_tokens=8192,initial_state=None,output_final_state=False,**kwargs):
 if segment_tokens<64 or segment_tokens%64:raise ValueError("segment_tokens must be multiple of64")
 if kwargs.get("cu_seqlens") is not None:raise ValueError("Packed inputs unsupported")
 out,state=SegmentedDelta.apply(q,k,v,g,beta,initial_state,kernel,segment_tokens,kwargs)
 return out,state if output_final_state else None
