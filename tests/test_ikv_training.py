"""Real MoT tests for the unchanged teacher-forced objective with IKV."""
from copy import deepcopy
import pytest
import torch
from n0_twam.models.ikv_training import run_ikv_training
from n0_twam.models.global_kv_retention import GlobalKVRetention
from n0_twam.models.model import FlexAttnFunc
from test_global_kv_retention import tiny_model

def case(batch=1):
    torch.manual_seed(18)
    # Four chronological phases: video+tactile 0, action 1, video+tactile 2, action 3.
    phases = torch.tensor([0,0,2,2, 0,0,2,2, 1,3,1,3, 0,2,0,2])
    clean = torch.tensor([0]*4+[1]*4+[0]*2+[1]*2+[0]*2+[1]*2).bool()
    kind = torch.tensor([0]*8+[1]*4+[2]*4)
    # Repeat samples within each modality/clean segment, matching N0-TWAM.
    splits=[4,4,2,2,2,2]
    def packed(x):
        return torch.cat([part.repeat(batch) for part in x.split(splits)])
    seq=torch.cat([torch.arange(batch).repeat_interleave(n) for n in splits])
    layout=dict(seq=seq,phase=packed(phases),clean=packed(clean),kind=packed(kind))
    n=len(seq);d=4
    h=torch.randn(1,n,d,requires_grad=True)
    text=torch.randn(1,batch*3,d)
    ts=torch.randn(1,n,6,d)
    temb=torch.randn(1,n,d)
    rope=torch.polar(torch.ones(1,n,1,2), torch.randn(1,n,1,2))
    rows=dict(world_time_id=layout["phase"].float()/2,
              grid_position=torch.zeros(n,3),kind=layout["kind"],
              observation_flag=torch.ones(n,dtype=torch.bool),duration=torch.ones(n),
              dino=torch.zeros(n,0),neoforce=torch.zeros(n,0),action=torch.zeros(n,3))
    return h,text,ts,temb,rope,dict(config=dict(capacity=32,retention=dict(top_k=3,seed=5)),layout=layout,rows=rows)

def dense_mask(layout,window=100):
    s,f,c=layout["seq"],layout["phase"],layout["clean"]
    causal=((c[:,None]&c[None,:])&(f[None,:]<=f[:,None])
            | ((~c[:,None]&c[None,:])&(f[None,:]<f[:,None])))
    same_noisy=(~c[:,None]&~c[None,:])&(f[:,None]==f[None,:])
    return ((s[:,None]==s[None,:]) & (causal|same_noisy)
            & ((f[:,None]-f[None,:]).abs()<=window))[None,None]

@pytest.mark.parametrize("batch",[1,2])
@pytest.mark.parametrize("checkpoint",[False,True])
def test_no_eviction_matches_original_outputs_and_gradients(batch,checkpoint):
    base=tiny_model(False).mot
    model=deepcopy(base)
    base.gradient_checkpointing=model.gradient_checkpointing=checkpoint
    h,text,ts,temb,rope,mem=case(batch)
    hv=h.detach().clone().requires_grad_()
    cross={}
    for name,start,end in [("video",0,8*batch),("action",8*batch,12*batch)]:
        query=mem["layout"]["seq"][start:end]
        text_seq=torch.arange(batch).repeat_interleave(3)
        cross[name]=(query[:,None]==text_seq[None,:])[None,None]
    base.set_masks(dense_self_mask=dense_mask(mem["layout"]),cross_masks=cross)
    expected=base(h,text,ts,temb,rope,[("video",0,8*batch),("action",8*batch,12*batch),("tactile",12*batch,16*batch)])
    actual=run_ikv_training(model,hv,text,ts,temb,rope,mem)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(hv.grad,h.grad,rtol=2e-4,atol=2e-6)
    for (_,p),(_,q) in zip(model.named_parameters(),base.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=3e-6)

def test_future_truth_cannot_change_past_output_or_retention(monkeypatch):
    model=tiny_model(False).mot
    args=case();h,text,ts,temb,rope,mem=args
    mem["config"]["capacity"]=4
    plans=[]
    original=GlobalKVRetention.plan
    def record(self,mask,count,rows):
        result=original(self,mask,count,rows)
        plans.append(tuple(x.clone() for x in result))
        return result
    monkeypatch.setattr(GlobalKVRetention,"plan",record)
    first=run_ikv_training(model,*args)
    initial=plans[:];plans.clear()
    h2=h.detach().clone()
    future=mem["layout"]["phase"]>=2
    h2[:,future]+=100
    second=run_ikv_training(model,h2,text,ts,temb,rope,mem)
    torch.testing.assert_close(first[:,~future],second[:,~future])
    for p,q in zip(initial[:2],plans[:2]):
        for a,b in zip(p,q):assert torch.equal(a,b)

def test_long_history_gradient_survives_and_capacity_is_enforced():
    model=tiny_model(False).mot
    model.gradient_checkpointing=True
    h,text,ts,temb,rope,mem=case()
    out=run_ikv_training(model,h,text,ts,temb,rope,mem)
    # Last noisy action can backprop through old clean video.
    out[:,9].square().sum().backward()
    assert h.grad[:,4:6].abs().sum()>0
    mem["config"]["capacity"]=1
    with pytest.raises(ValueError,match="exceeds"):
        run_ikv_training(model,h,text,ts,temb,rope,mem)

def test_query_usage_is_all_layer_and_policy_is_reset_per_sample(monkeypatch):
    model=tiny_model(False).mot
    args=case()
    counts=[]
    original=GlobalKVRetention.add_usage
    def record(self,measurements):
        counts.append(len(measurements));return original(self,measurements)
    monkeypatch.setattr(GlobalKVRetention,"add_usage",record)
    a=run_ikv_training(model,*args)
    b=run_ikv_training(model,*args)
    torch.testing.assert_close(a,b)
    assert counts and all(n==model.num_layers for n in counts)


def test_motion_support_is_causal_and_keeps_observed_branch():
    from n0_twam.models.motion_training import prepare_causal_motion
    indices=torch.tensor([[[0,1],[1,2],[2,3],[3,0]]])
    grid=torch.zeros(1,4,16)
    grid[:,0]=torch.arange(4).repeat_interleave(4)
    original=dict(rgb_motion_patch_indices=indices,
                  rgb_motion_valid_mask=torch.ones_like(indices,dtype=torch.bool),grid_id=grid)
    changed=deepcopy(original);changed["rgb_motion_patch_indices"][:,2:]=0
    prepare_causal_motion(original,2);prepare_causal_motion(changed,2)
    assert (original["rgb_motion_patch_indices"][:,:2]==-1).all()
    assert original["rgb_motion_patch_indices"][0,2:].tolist()==[[1,2],[1,2]]
    assert torch.equal(original["condition_motion"]["rgb_motion_patch_indices"],indices)
    assert torch.equal(original["rgb_motion_patch_indices"],changed["rgb_motion_patch_indices"])

def test_sparse_noisy_and_clean_validity_are_independent():
    shape=(1,2,2,4,4)
    noisy=torch.tensor([[False,False,True,False]])
    observed=torch.tensor([[True,False,True,True]])
    layout=FlexAttnFunc.init_mask(shape,(1,3,2,2,1),0,1,100,(1,2,2),"cpu",
                                  latent_token_frame_ids=torch.tensor([[0,0,1,1]]),
                                  latent_token_valid_mask=noisy,
                                  latent_condition_valid_mask=observed,return_layout=True)
    assert (layout["seq"][:4]>=0).tolist()==noisy[0].tolist()
    assert (layout["seq"][4:8]>=0).tolist()==observed[0].tolist()

def test_same_phase_clean_features_do_not_select_noisy_history():
    model=tiny_model(False).mot
    h,text,ts,temb,rope,mem=case()
    mem["config"]["capacity"]=4
    # Video phase 2 has different incoming semantic information in two examples.
    mem["rows"]["dino"]=torch.randn(len(mem["layout"]["seq"]),3)
    altered=deepcopy(mem)
    target_clean=(mem["layout"]["phase"]==2)&mem["layout"]["clean"]
    altered["rows"]["dino"][target_clean]*=-100
    h2=h.detach().clone();h2[:,target_clean]+=99
    a=run_ikv_training(model,h,text,ts,temb,rope,mem)
    b=run_ikv_training(model,h2,text,ts,temb,rope,altered)
    target_noisy=(mem["layout"]["phase"]==2)&~mem["layout"]["clean"]
    torch.testing.assert_close(a[:,target_noisy],b[:,target_noisy],rtol=1e-5,atol=2e-6)


def training_input(device="cpu",motion=False):
    from n0_twam.utils.utils import get_mesh_id
    from n0_twam.models.motion_training import prepare_causal_motion
    torch.manual_seed(45)
    def stream(c,h,w,action=False):
        x=torch.randn(1,c,3,h,w,device=device,dtype=torch.bfloat16)
        grid=get_mesh_id(3,h if action else h//2,w if action else w//2,int(action),
                         action=action).to(device)[None]
        return dict(noisy_latents=x,latent=torch.randn_like(x),targets=torch.randn_like(x),
                    timesteps=torch.full((1,3),500.,device=device),
                    cond_timesteps=torch.zeros(1,3,device=device),grid_id=grid,
                    text_emb=torch.randn(1,3,8,device=device,dtype=torch.bfloat16))
    v=stream(2,4,4);a=stream(3,2,1,True)
    a["actions_mask"]=torch.ones_like(a["latent"])
    for key in ("tactile_global_noisy_latent","tactile_global_clean_latent","tactile_global_targets"):
        a[key]=torch.randn(1,1,48,3,4,4,device=device,dtype=torch.bfloat16)
    a["tactile_sensor_ids"]=torch.zeros(1,1,dtype=torch.long,device=device)
    a["tactile_global_timesteps"]=a["timesteps"]
    a["tactile_global_cond_timesteps"]=a["cond_timesteps"]
    if motion:
        v["rgb_motion_patch_indices"]=torch.tensor([[[0,-1],[1,2],[0,3]]],device=device)
        v["rgb_motion_valid_mask"]=v["rgb_motion_patch_indices"]>=0
        prepare_causal_motion(v,1)
    return dict(latent_dict=v,action_dict=a,chunk_size=1,window_size=100)

def original_loss():
    from types import SimpleNamespace
    from test_rgb_motion_model_integration import _load_trainer_class
    from n0_twam.utils.scheduler import FlowMatchScheduler
    cls=_load_trainer_class()
    trainer=cls.__new__(cls)
    trainer.patch_size=(1,2,2)
    trainer.config=SimpleNamespace(tactile_diffusion_loss_weight=1)
    trainer.gradient_accumulation_steps=1
    trainer.train_scheduler_latent=FlowMatchScheduler()
    trainer.train_scheduler_action=FlowMatchScheduler()
    for s in (trainer.train_scheduler_latent,trainer.train_scheduler_action):
        s.set_timesteps(1000,training=True)
    return trainer

@pytest.mark.parametrize("motion,ikv",[(False,False),(True,False),(False,True),(True,True)])
def test_full_original_train_and_loss_all_modes(monkeypatch,motion,ikv):
    from types import SimpleNamespace
    from n0_twam.models.model import custom_sdpa
    # Use original mask_mod in an eager backend; no Flex compilation needed for
    # CPU regression. A separate CUDA smoke exercises the real compiled path.
    def make(mask,*args,**kwargs):
        return SimpleNamespace(mask_mod=mask)
    def attention(self,q,k,v):
        if self.block_mask is None:
            return custom_sdpa(q,k,v)
        mod=self.block_mask.mask_mod
        mask=mod(torch.tensor(0),torch.tensor(0),torch.arange(q.shape[1])[:,None],torch.arange(k.shape[1])[None,:])
        return custom_sdpa(q,k,v,mask[None,None])
    monkeypatch.setattr(FlexAttnFunc,"compiled_create_block_mask",staticmethod(make))
    monkeypatch.setattr(FlexAttnFunc,"forward",attention)
    model=tiny_model(False).to(torch.bfloat16).train()
    model.use_rgb_motion_tokens=motion;model.rgb_motion_require_index=False
    model.mot.gradient_checkpointing=True
    data=training_input(motion=motion)
    if ikv:data["ikv_training"]=dict(capacity=12,retention=dict(top_k=2))
    outputs=model(data,train_mode=True)
    losses=original_loss().compute_loss(data,outputs)
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"])
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0


def test_semantic_motion_joint_training_and_tactile_only_support():
    model=tiny_model(False).to(torch.bfloat16).train()
    model.use_rgb_motion_tokens=True
    model.rgb_motion_require_index=True
    data=training_input(motion=True)
    latent=data['latent_dict']
    latent.update(latent.pop('condition_motion'))
    idx=latent['rgb_motion_patch_indices'];valid=idx>=0
    visual=valid.clone();visual[:,1,0]=False
    latent.update(world_time_id=torch.arange(3)[None,:,None].expand_as(idx).clone(),
                  dino_features=torch.randn(*idx.shape,2),
                  neoforce_features=torch.randn(*idx.shape,2),
                  observation_flag=valid.long(),visual_valid=visual,
                  tactile_valid=valid & ~visual)
    from n0_twam.models.motion_training import prepare_causal_motion
    prepare_causal_motion(latent,1)
    assert not latent['rgb_motion_valid_mask'][0,2,0]
    assert latent['condition_motion']['rgb_motion_valid_mask'][0,1,0]
    data['ikv_training']=dict(capacity=12,retention=dict(top_k=2))
    output=model(data,train_mode=True)
    loss=original_loss().compute_loss(data,output)['total_loss']
    loss.backward()
    assert torch.isfinite(loss)


def test_condition_mask_uses_own_frame_addresses():
    layout=FlexAttnFunc.init_mask((1,2,3,4,4),(1,3,3,2,1),0,1,100,(1,2,2),'cpu',
        latent_token_frame_ids=torch.tensor([[0,0,1,1]]),
        latent_condition_frame_ids=torch.tensor([[1,1,2,2]]),return_layout=True)
    assert layout['phase'][:8].tolist()==[0,0,2,2,2,2,4,4]


@torch.no_grad()
def test_sparse_global_serving_compacts_padding_and_uses_selected_features():
    from test_global_kv_retention import model_input,cache_snapshot,assert_cache_equal
    model=tiny_model(False)
    model.use_rgb_motion_tokens=True
    model.rgb_motion_require_index=True
    model.configure_global_retention('test',top_k=3)
    for time in range(5):
        data=model_input(time)
        data.update(rgb_motion_patch_indices=torch.tensor([[[1,-1,3]]]),
                    rgb_motion_valid_mask=torch.tensor([[[True,False,True]]]),
                    world_time_id=torch.full((1,1,3),time),
                    dino_features=torch.tensor([[[[1.,2.],[999.,999.],[3.,4.]]]]),
                    neoforce_features=torch.empty(1,1,3,0),
                    observation_flag=torch.ones(1,1,3,dtype=torch.long),
                    visual_valid=torch.tensor([[[True,False,True]]]),
                    tactile_valid=torch.zeros(1,1,3,dtype=torch.bool))
        output=model(data,update_cache=2,cache_name='test')
        assert all(torch.isfinite(x).all() for x in output)
    state=model.get_global_retention('test')
    assert len(state['token_uid'])==24
    assert state['t0']==4
    assert not (state['dino']==999).any()
    assert ((state['dino']==torch.tensor([1.,2.])).all(-1)).any()
    before=cache_snapshot(model)
    model(data,update_cache=0,cache_name='test')
    assert_cache_equal(before,cache_snapshot(model))


def test_dense_sidecar_alignment_and_temporal_crop(tmp_path):
    # This standalone reader needs no optional LeRobot package.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('ikv_index_test',
        Path(__file__).parents[1]/'n0_twam/dataset/ikv_index.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    load_dense_index=module.load_dense_index
    path=tmp_path/'index.pth'
    features=torch.arange(24.).reshape(3,4,2)
    payload=dict(camera_keys=['top','wrist'],patch_size=[1,2,2],
                 spatial_grid_shape=[2,2],frame_ids=[0,4,8],dino_features=features)
    torch.save(payload,path)
    args=dict(camera_keys=['top','wrist'],patch_size=(1,2,2),grid_shape=(2,2),
              latent_frame_ids=[0,4,8],full_frames=3,start=1,end=3)
    actual=load_dense_index(path,**args)
    torch.testing.assert_close(actual['dino_features'],features[1:3])
    assert actual['neoforce_features'].shape==(2,4,0)
    with pytest.raises(ValueError,match='camera_keys'):
        load_dense_index(path,**dict(args,camera_keys=['wrist','top']))
    with pytest.raises(ValueError,match='frame_ids'):
        load_dense_index(path,**dict(args,latent_frame_ids=[0,3,7]))
    payload['dino_features'][0,0,0]=float('nan');torch.save(payload,path)
    with pytest.raises(ValueError,match='finite'):
        load_dense_index(path,**args)

@pytest.mark.parametrize('chunk_size',[2,4])
def test_motion_cold_chunk_has_finite_original_loss(chunk_size):
    from n0_twam.models.motion_training import prepare_causal_motion
    model=tiny_model(False).to(torch.bfloat16).train()
    model.use_rgb_motion_tokens=True;model.rgb_motion_require_index=False
    data=training_input(motion=True)
    latent=data['latent_dict'];latent.update(latent.pop('condition_motion'))
    prepare_causal_motion(latent,chunk_size)
    data['chunk_size']=chunk_size
    data['ikv_training']=dict(capacity=32)
    output=model(data,train_mode=True)
    loss=original_loss().compute_loss(data,output)['total_loss']
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize('override,match',[
    ({'use_mot':False},'use_mot'),
    ({'ikv_train_capacity':0},'positive integer'),
    ({'ikv_train_capacity':True},'positive integer'),
])
def test_training_config_rejects_unusable_ikv(override,match):
    from types import SimpleNamespace
    from test_rgb_motion_model_integration import Trainer
    config=dict(use_mot=True,use_ikv_training=True,ikv_train_capacity=32)
    config.update(override)
    with pytest.raises(ValueError,match=match):
        Trainer._validate_rgb_motion_training_config(SimpleNamespace(**config))


def test_serve_rejects_ikv_checkpoint_capacity_and_policy_mismatch(tmp_path):
    from test_rgb_motion_server_helpers import _consistency_server
    metadata=dict(use_ikv_training=True,ikv_train_capacity=32,kv_retention={'top_k':2})
    server=_consistency_server(tmp_path,enabled=False,live_cameras=['top'],metadata=metadata)
    config=server.job_config
    config.use_ikv_training=True;config.ikv_train_capacity=32
    config.kv_cache_policy='global';config.kv_retention={'top_k':2}
    server._check_train_serve_consistency()
    config.ikv_train_capacity=31
    with pytest.raises(RuntimeError,match='ikv_train_capacity'):
        server._check_train_serve_consistency()
    config.ikv_train_capacity=32;config.kv_cache_policy='fifo'
    with pytest.raises(RuntimeError,match='global retention'):
        server._check_train_serve_consistency()
