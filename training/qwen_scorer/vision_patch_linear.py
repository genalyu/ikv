"""Exact algebraic replacement of a non-overlapping full-patch Conv3d by GEMM.
Keeps original parameter objects, shapes, names, and checkpoint compatibility.
"""
import torch
from torch.nn import functional as F

def enable_linear_patch_embed(visual):
 patch=visual.patch_embed
 if getattr(patch,'_linear_patch_enabled',False):return
 proj=patch.proj
 assert tuple(proj.kernel_size)==tuple(proj.stride)
 assert tuple(proj.padding)==(0,0,0) and tuple(proj.dilation)==(1,1,1) and proj.groups==1
 width=proj.weight[0].numel()
 def forward(hidden_states):
  assert hidden_states.numel()%width==0
  return F.linear(hidden_states.reshape(-1,width).to(proj.weight.dtype),proj.weight.flatten(1),proj.bias)
 patch.forward=forward
 patch._linear_patch_enabled=True
