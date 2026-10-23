"""CPU-tested scheduling candidate only; NOT imported by current training."""
from full_history_sampler import FullHistoryLengthSampler
def estimated_cost(n,cost_model=None):
 z=n/50000
 if cost_model is not None:
  tier="0" if n<=100000 else "1" if n<=200000 else "2"
  a,b,c=cost_model["compute_coefficients"][tier];u,v=cost_model.get("collation_coefficients",[0.,0.])
  return a+b*z+c*z*z+u+v*z
 return (z+.382*z*z)*(1.15 if n>100000 else 1)*(1.8 if n>200000 else 1)
def balanced_group(indices,lengths,world_size,cost_model=None):
 assert len(indices)%world_size==0
 capacity=len(indices)//world_size;loads=[0.]*world_size;parts=[[] for _ in range(world_size)]
 for index in sorted(indices,key=lambda i:estimated_cost(lengths[i],cost_model),reverse=True):
  rank=min((r for r in range(world_size) if len(parts[r])<capacity),key=lambda r:loads[r])
  parts[rank].append(index);loads[rank]+=estimated_cost(lengths[index],cost_model)
 return [index for row in zip(*parts) for index in row]
class CostBalancedLengthSampler(FullHistoryLengthSampler):
 def __init__(self,*args,world_size,cost_model=None,**kwargs):
  super().__init__(*args,**kwargs);self.world_size=world_size;self.cost_model=cost_model
 def __iter__(self):
  original=list(super().__iter__());result=[]
  for start in range(0,len(original),self.batch_size):
   result.extend(balanced_group(original[start:start+self.batch_size],self.lengths,self.world_size,self.cost_model))
  return iter(result)
