"""CUDA A/B for direct retention groups and shared index packs, including AdamW."""
from copy import deepcopy
import sys,json
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from n0_twam.models.mot import WanMoTTransformer3DModel
import n0_twam.models.ikv_training as ikv
import n0_twam.models.mot as mot
from n0_twam.models.compact_ikv_attention import build_compact_ikv_groups
from test_ikv_training import training_input,original_loss

torch.set_num_threads(1);torch.manual_seed(123)
a=WanMoTTransformer3DModel(patch_size=(1,2,2),num_attention_heads=1,attention_head_dim=32,in_channels=2,out_channels=2,action_dim=3,text_dim=8,freq_dim=4,ffn_dim=32,num_layers=2,rope_max_seq_len=32,attn_mode="torch",use_local_tactile=False,use_rgb_motion_tokens=False).to(device="cuda",dtype=torch.bfloat16).train()
b=deepcopy(a)
a.mot.gradient_checkpointing=b.mot.gradient_checkpointing=True
raw=training_input("cuda",False)
raw["latent_dict"]["dense_dino_features"]=torch.ones(1,3,4,2,device="cuda")
raw["ikv_training"]=dict(capacity=12,retention=dict(version=2,video_capacity=4,action_capacity=4,tactile_capacity=4,query_weight=0.,action_query_weight=0.,tactile_query_weight=0.),execution="masked",sample_capacity=False,compact_attention=True,compact_max_packed_keys=512)
original=ikv.build_ikv_support_plan
def legacy(memory,device,*,return_groups=False):
 plan=original(memory,device)
 return dict(groups=build_compact_ikv_groups(memory["layout"],plan)) if return_groups else plan
set_masks=a.mot.set_masks
def unpacked(*args,**kwargs):
 set_masks(*args,**kwargs)
 for layer in a.mot.shared_attn:layer.compact_chunks=None
a.mot.set_masks=unpacked
ikv.build_ikv_support_plan=legacy
out=a(deepcopy(raw),train_mode=True);loss=original_loss().compute_loss(raw,out)["total_loss"];loss.backward()
ikv.build_ikv_support_plan=original
counter=[0];pack=mot._pack_group_chunks
def counted(*args,**kwargs):counter[0]+=1;return pack(*args,**kwargs)
mot._pack_group_chunks=counted
candidate=b(deepcopy(raw),train_mode=True);closs=original_loss().compute_loss(raw,candidate)["total_loss"];closs.backward()
assert counter[0]==1,counter
assert b.mot.shared_attn[0].compact_chunks is b.mot.shared_attn[1].compact_chunks
torch.testing.assert_close(loss,closs,rtol=0,atol=0)
checked=0
for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()):
 if x.grad is None:assert y.grad is None;continue
 assert torch.isfinite(y.grad).all(),name
 torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0,msg=lambda m:name+":"+m);checked+=1
opts=[torch.optim.AdamW(m.parameters(),lr=1e-4,fused=True,foreach=False) for m in (a,b)]
for opt in opts:opt.step()
for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
print(json.dumps(dict(status="passed",loss=float(loss.detach()),parameter_gradients_checked=checked,pack_calls=counter[0],optimizer_parameters_equal=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
