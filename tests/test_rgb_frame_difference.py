import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('rgbdiff',Path(__file__).parents[1]/'n0_twam/preprocessing/rgb_frame_difference.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def run(frames, previous=None, anchors=None):
    keys=tuple(frames)
    seq,_=m.parse_rgb_cameras({'camera_keys':keys,'cameras':{k:{'rgb':v} for k,v in frames.items()}},keys)
    anchors=anchors or [len(next(iter(frames.values())))-1]
    return m.RGBFrameDifferencePreprocessor(camera_keys=keys,height=64,width=64)(seq,anchor_indices=anchors,world_time_ids=list(range(len(anchors))),previous_frames=previous)

def test_first_all_then_static():
    x=torch.zeros(1,64,64,3,dtype=torch.uint8)
    assert run({'a':x})['motion_valid_mask'].sum()==4
    assert run({'a':x},{'a':{'rgb':x}})['motion_valid_mask'].sum()==0

def test_same_camera_and_width_concat():
    x=torch.zeros(1,64,64,3,dtype=torch.uint8);y=x.clone();y[:,:32,:32]=255
    out=run({'a':x,'b':y},{'a':{'rgb':x},'b':{'rgb':x}})
    assert out['motion_indices'][out['motion_valid_mask']].tolist()==[2]

def test_transient_motion_not_lost():
    x=torch.zeros(4,64,64,3,dtype=torch.uint8);x[1,:32,:32]=255
    out=run({'a':x},{'a':{'rgb':x[:1]}},[3])
    assert out['motion_valid_mask'].sum()==1

def test_input_validation():
    import pytest
    with pytest.raises(ValueError):run({'a':torch.full((1,64,64,3),float('nan'))})
    with pytest.raises(ValueError):run({'a':torch.zeros(1,64,64,3)}, {})

def test_reset_full_again():
    x=torch.zeros(1,64,64,3,dtype=torch.uint8)
    assert run({'a':x},None)['motion_valid_mask'].all()
