"""Bound FLA invocation length; carry recurrent state and its gradients across all segments."""
import functools,json,os,time
import torch

def segmented_call(kernel,q,k,v,g,beta,*,segment_tokens=8192,initial_state=None,output_final_state=False,**kwargs):
 if segment_tokens<64 or segment_tokens%64:raise ValueError("segment_tokens must be a positive multiple of FLA chunk size 64")
 if kwargs.get("cu_seqlens") is not None:raise ValueError("Packed variable-length inputs require a separate implementation")
 state=initial_state; outputs=[]
 for start in range(0,q.shape[1],segment_tokens):
  end=min(q.shape[1],start+segment_tokens)
  out,state=kernel(q[:,start:end].contiguous(),k[:,start:end].contiguous(),v[:,start:end].contiguous(),g=g[:,start:end].contiguous(),beta=beta[:,start:end].contiguous(),initial_state=state,output_final_state=True,**kwargs)
  if state is None:raise RuntimeError("Kernel did not return carry state")
  # Do not detach carry: training must retain dependencies on all historical tokens.
  if not torch.isfinite(out).all() or not torch.isfinite(state).all():
   raise FloatingPointError(f"Nonfinite segmented delta output/state: tokens [{start},{end})")
  outputs.append(out)
 return torch.cat(outputs,dim=1),state if output_final_state else None

def enable_segmented_delta(model,segment_tokens=8192,bounded_backward=False):
 from bounded_delta import bounded_segmented_call
 call=bounded_segmented_call if bounded_backward else segmented_call
 count=0
 for name,module in model.named_modules():
  if type(module).__name__!="Qwen3_5GatedDeltaNet":continue
  module.chunk_gated_delta_rule=functools.partial(call,module.chunk_gated_delta_rule,segment_tokens=segment_tokens)
  count+=1
 print(json.dumps(dict(event="segmented_delta_enabled",layers=count,segment_tokens=segment_tokens,rank=os.environ.get("RANK"),time=time.time())),flush=True)
 return count
