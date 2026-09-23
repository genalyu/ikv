"""Exercise actual server helper bodies without loading GPU model dependencies."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import numpy as np
import torch


def test_rgb_server_roundtrip(monkeypatch):
    root=Path(__file__).parents[1]
    name='n0_twam.preprocessing.rgb_frame_difference'
    spec=importlib.util.spec_from_file_location(name,root/'n0_twam/preprocessing/rgb_frame_difference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules,name,module)
    tree=ast.parse((root/'n0_twam/n0_twam_server.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_rgb_motion_from_raw_inputs' for m in n.body))
    methods=[m for m in cls.body if isinstance(m,ast.FunctionDef) and (m.name.startswith('_rgb_') or m.name in ('_validate_rgb_motion_server_config','_get_rgb_motion_preprocessor','_validate_rgb_motion_payload_provenance','_validate_rgb_motion_observation_binding'))]
    code=ast.Module(body=[ast.ClassDef(name='Server',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ns={'torch':torch,'np':np};exec(compile(ast.fix_missing_locations(code),'<server helpers>','exec'),ns)
    s=ns['Server']();s._get_kv_dino_encoder=lambda: (lambda x: torch.ones(len(x),2,2,8));s.device=torch.device('cpu');s.height=s.width=64
    s.job_config=SimpleNamespace(rgb_motion_input_mode='rgb',use_rgb_motion_tokens=True,rgb_motion_online_preprocess=True,patch_size=(1,2,2),height=64,width=64,obs_cam_keys=['camera'])
    s._validate_rgb_motion_server_config(s.job_config)
    s.streaming_vae=SimpleNamespace(feat_cache=[None],vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)))
    image=torch.zeros(64,64,3,dtype=torch.uint8)
    first=s._rgb_motion_for_frames({'obs':{'camera':image}},1,0,observed=True,streaming_vae_warm=False)
    assert first['motion_valid_mask'].sum()==4
    obs={'obs':[{'camera':image} for _ in range(4)]}
    assert s._rgb_motion_observed_frame_count(obs,streaming_vae_warm=True)==1
    second=s._rgb_motion_for_frames(obs,1,1,observed=True,streaming_vae_warm=True)
    assert second['motion_valid_mask'].sum()==0
