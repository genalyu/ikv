"""Release unused PyTorch cache before cold Triton autotuning, preserving kernel math."""
import functools,torch
def enable_autotune_memory_headroom():
 from triton.runtime.autotuner import Autotuner
 if getattr(Autotuner._bench,"_ikv_headroom",False):return
 original=Autotuner._bench
 original_disk=Autotuner.check_disk_cache
 @functools.wraps(original_disk)
 def disk(self,*args,**kwargs):
  # A cached tuning result can still require a cold CUDA module load.
  if torch.cuda.is_available():torch.cuda.empty_cache()
  return original_disk(self,*args,**kwargs)
 Autotuner.check_disk_cache=disk
 @functools.wraps(original)
 def bench(self,*args,**kwargs):
  # Driver module loading and benchmark buffers bypass the PyTorch allocator.
  # Emptying only unused cached blocks exposes memory to these allocations.
  if torch.cuda.is_available():torch.cuda.empty_cache()
  return original(self,*args,**kwargs)
 bench._ikv_headroom=True
 Autotuner._bench=bench
