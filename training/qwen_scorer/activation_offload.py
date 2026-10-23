"""Offload activations without duplicating resident parameters or caching pinned buffers."""
import json,os,time,threading,weakref
from pathlib import Path
import torch
class Packed:
 pass
class ActivationOffload:
 def __init__(self,model,gpu_budget_gib=0,backward_gpu_budget_gib=0,deduplicate=True):
  self.deduplicate=deduplicate;self.cache={};self.cache_hits=0
  self.device=next(p.device for p in model.parameters() if p.device.type=='cuda')
  self.parameters={p.untyped_storage().data_ptr() for p in model.parameters() if p.device.type=='cuda'}
  self.backward_gpu_budget_bytes=int(backward_gpu_budget_gib*2**30);self.gpu_budget_bytes=int(gpu_budget_gib*2**30);self.gpu_live_bytes=0
  self.live_bytes=0;self.peak_bytes=0;self.parameter_hits=0;self.phase='idle'
 def release(self,n):self.live_bytes-=n
 def release_gpu(self,n):self.gpu_live_bytes-=n
 def remember(self,t,p,key):
  if self.deduplicate:
   def cleanup(ref):
    entry=self.cache.get(key)
    if entry is not None and entry[1] is ref:self.cache.pop(key,None)
   self.cache[key]=(weakref.ref(t),weakref.ref(p,cleanup))
  return p
 def pack(self,t):
  if t.device.type!='cuda':return t
  if t.untyped_storage().data_ptr() in self.parameters:
   self.parameter_hits+=1
   return t
  key=(t.device,t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),t.dtype,t._version)
  entry=self.cache.get(key) if self.deduplicate else None
  if entry is not None:
   source,p=entry[0](),entry[1]()
   if source is not None and p is not None and source._version==t._version:
    self.cache_hits+=1
    return p
  n=t.numel()*t.element_size()
  budget=self.gpu_budget_bytes+(self.backward_gpu_budget_bytes if self.phase=="backward" else 0)
  allow=self.phase=="forward" or (self.phase=="backward" and torch.cuda.memory_allocated(self.device)<58*2**30)
  if allow and self.gpu_live_bytes+n<=budget and t.untyped_storage().nbytes()<=n:
   p=Packed();p.device=t.device;p.tensor=t.detach();self.gpu_live_bytes+=n
   weakref.finalize(p,self.release_gpu,n)
   return self.remember(t,p,key)
  p=Packed();p.device=t.device;p.tensor=t.detach().to('cpu',non_blocking=False)
  n=p.tensor.numel()*p.tensor.element_size();self.live_bytes+=n;self.peak_bytes=max(self.peak_bytes,self.live_bytes)
  weakref.finalize(p,self.release,n)
  return self.remember(t,p,key)
 def unpack(self,p):
  if isinstance(p,Packed):return p.tensor.to(p.device,non_blocking=False)
  return p
 def context(self):return torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack)
 def report(self):
  row=dict(event='memory_status',rank=os.environ.get('RANK'),phase=self.phase,time=time.time(),offload_cache_hits=self.cache_hits,retained_gpu_gib=self.gpu_live_bytes/2**30,offload_live_gib=self.live_bytes/2**30,offload_peak_gib=self.peak_bytes/2**30,parameter_copy_avoided=self.parameter_hits)
  if torch.cuda.is_available():
   row.update(gpu_allocated_gib=torch.cuda.memory_allocated(self.device)/2**30,gpu_reserved_gib=torch.cuda.memory_reserved(self.device)/2**30,gpu_peak_gib=torch.cuda.max_memory_allocated(self.device)/2**30)
  for name in ['memory.current','memory.peak','memory.max','memory.events']:
   p=Path('/sys/fs/cgroup')/name
   if p.exists():row['cgroup_'+name]=p.read_text().strip()
  for l in Path('/proc/self/status').read_text().splitlines():
   if l.startswith(('VmRSS:','VmLck:','VmPin:')):row[l.split(':')[0]]=l.split(':',1)[1].strip()
  print(json.dumps(row),flush=True)
 def start_monitor(self):
  def loop():
   while True:
    self.report();time.sleep(20)
  threading.Thread(target=loop,daemon=True).start()

def gpu_equivalence_test():
 torch.manual_seed(121)
 m=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.SiLU(),torch.nn.Linear(64,8)).cuda()
 x=torch.randn(16,32,device='cuda',requires_grad=True)
 def run(ctx):
  m.zero_grad(set_to_none=True);x.grad=None
  with ctx:
   y=m(x);y.square().sum().backward()
  return y.detach().clone(),[x.grad.clone()]+[p.grad.clone() for p in m.parameters()]
 import contextlib
 a,ga=run(contextlib.nullcontext());off=ActivationOffload(m,gpu_budget_gib=.000005);off.phase='forward';b,gb=run(off.context())
 torch.testing.assert_close(a,b,rtol=0,atol=0)
 for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=0,atol=0)
 assert off.parameter_hits>0
 print(json.dumps(dict(event='activation_offload_gpu_equivalence_passed',peak_bytes=off.peak_bytes,parameter_hits=off.parameter_hits)),flush=True)
if __name__=='__main__':gpu_equivalence_test()
