import json
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import torch
from n0_twam.models.ikv_training import training_metadata
from n0_twam.dataset.ikv_index import load_dense_index
from n0_twam.task_pipeline.features import build_tactile_payload
from test_neosim_explicit_arms import make_task

def test_tactile_metadata_is_sensor_major_and_uses_measured_durations():
    f,s,p,d=2,2,16,3
    tactile=torch.arange(f*s*p*d).reshape(1,f,s,p,d).float()
    times=torch.cat((torch.arange(2),torch.arange(2),torch.arange(2).repeat_interleave(p).repeat(s))).float()
    n=len(times);grid=torch.zeros(1,4,n);grid[0,0]=times
    layout=dict(seq=torch.zeros(n,dtype=torch.long),kind=torch.tensor([0]*2+[1]*2+[2]*(s*f*p)),clean=torch.ones(n,dtype=torch.bool))
    latent=dict(dense_dino_features=torch.ones(1,2,1,3),tactile_dino_features=tactile,
        frame_contact_duration=torch.tensor([[0.,.5]]),
        tactile_contact_duration=torch.tensor([[[0.,.25],[.5,0.]]]))
    action=dict(latent=torch.zeros(1,10,2,1,1))
    rows=training_metadata(grid,layout,[0,2,0,2,0,64,0],latent,action,None,version=2)
    torch.testing.assert_close(rows["neoforce"][4:],tactile[0].permute(1,0,2,3).reshape(-1,3))
    torch.testing.assert_close(rows["contact_duration"][:2],torch.tensor([0.,.5]))
    torch.testing.assert_close(rows["contact_duration"][4:],torch.tensor([0.]*16+[.5]*16+[.25]*16+[0.]*16))
    assert rows["neoforce"][:4].eq(0).all()
    assert rows["contact_present"].equal(rows["contact_duration"]>0)

def payload():
    return dict(camera_keys=["top"],patch_size=[1,2,2],spatial_grid_shape=[1,1],frame_ids=[0,1,2,3,4],
        dino_features=torch.ones(2,1,3),neoforce_features=torch.empty(2,1,0),
        tactile_camera_keys=["left","right"],tactile_dino_features=torch.ones(2,2,16,3),
        frame_contact_duration=torch.tensor([0.,.5]),tactile_contact_duration=torch.tensor([[0.,.2],[.5,0.]]))

def load(path,**kw):
    return load_dense_index(path,camera_keys=["top"],patch_size=[1,2,2],grid_shape=[1,1],
        latent_frame_ids=[0,1,2,3,4],full_frames=2,**kw)

def test_tactile_sidecar_slice_and_sensor_contract(tmp_path):
    path=tmp_path/"index.pth";torch.save(payload(),path)
    x=load(path,start=1,end=2,tactile_keys=["left","right"])
    assert x["tactile_dino_features"].shape==(1,2,16,3)
    with pytest.raises(ValueError,match="sensor order"):load(path,tactile_keys=["right","left"])
    v=payload();v["tactile_contact_duration"][0,0]=-1;path=tmp_path/"negative.pth";torch.save(v,path)
    with pytest.raises(ValueError,match="duration"):load(path)

def test_builder_uses_native_tactile_rgb_and_force_clock(tmp_path,monkeypatch):
    source=tmp_path/"raw";source.mkdir();task=make_task(source,1)
    task["source"]=str(source)
    with h5py.File(source/"hdf5/0.hdf5","a") as h:
        force=np.zeros((6,169,3));force[1:,0,0]=.01;h["tactile/left_tactile/contact_force"]=force
    root=tmp_path/"dataset";(root/"conversion_episodes").mkdir(parents=True)
    (root/"conversion_episodes/episode_000000.json").write_text(json.dumps(dict(source_id="0")))
    seen=[]
    def encoder(rgb):
        seen.append(rgb.clone())
        return SimpleNamespace(tokens=torch.ones(1,16,16,3))
    def frames(path,ids):
        for _ in ids:yield np.tile(np.array([11,22,33],dtype=np.uint8),(256,256,1))
    monkeypatch.setattr("n0_twam.task_pipeline.features.selected_rgb",frames)
    x=build_tactile_payload(task,root,"chunk-000","episode_000000_0_5.pth",[0,1,2,3,4],[0,4],encoder)
    assert x["tactile_dino_features"].shape==(2,1,16,3)
    assert seen[0][0,0,0].tolist()==[11,22,33]
    torch.testing.assert_close(x["frame_contact_duration"],torch.tensor([0.,4/12]))
