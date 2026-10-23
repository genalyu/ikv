"""Fail at the first non-finite module boundary; never repair values silently."""
import json,os,time
import torch

def check_finite(value,name):
 if isinstance(value,torch.Tensor) and value.is_floating_point():
  with torch.no_grad():
   parts=value.detach().split(1024,dim=1) if value.ndim>=3 else (value.detach(),)
   flags=torch.stack([torch.isfinite(part).all() for part in parts])
   if not flags.all():
    i=int((~flags).nonzero(as_tuple=False)[0,0])
    raise FloatingPointError(f"First non-finite tensor at {name}, slice={i}, shape={tuple(value.shape)}, dtype={value.dtype}")
 elif isinstance(value,(tuple,list)):
  for i,item in enumerate(value):check_finite(item,f"{name}[{i}]")
 elif isinstance(value,dict):
  for key,item in value.items():check_finite(item,f"{name}.{key}")

def install_numerics_checks(model):
 handles=[]
 for name,module in model.named_modules():
  cls=type(module).__name__
  if cls in {"Qwen3_5DecoderLayer","Qwen3_5GatedDeltaNet","Qwen3_5Attention","Qwen3_5RMSNormGated","FusedRMSNormGated","Qwen3_5VisionModel"}:
   def before(mod,args,label=name):check_finite(args,label+".input")
   def after(mod,args,result,label=name):
    if hasattr(result,"pooler_output"):check_finite(result.pooler_output,label+".pooler_output")
    else:check_finite(result,label+".output")
    print(json.dumps(dict(event="finite_module",module=label,rank=os.environ.get("RANK"),time=time.time())),flush=True)
   handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
 return handles

def install_kernel_checks(model):
 originals=[]
 # Check actual kernel boundaries, including kwargs that module hooks do not see.
 for name,module in model.named_modules():
  if type(module).__name__!="Qwen3_5GatedDeltaNet":continue
  for attr in ("causal_conv1d_fn","chunk_gated_delta_rule"):
   original=getattr(module,attr,None)
   if original is None:continue
   def checked(*args,_fn=original,_label=name+"."+attr,**kwargs):
    check_finite(args,_label+".args");check_finite(kwargs,_label+".kwargs")
    result=_fn(*args,**kwargs)
    check_finite(result,_label+".result")
    return result
   originals.append((module,attr,original))
   setattr(module,attr,checked)
 return originals

def remove_numerics_checks(handles,originals):
 """Remove detailed module/kernel wrappers; standalone critical guards stay active."""
 for handle in handles:handle.remove()
 for module,attr,original in originals:setattr(module,attr,original)
